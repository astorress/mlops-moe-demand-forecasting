"""
Clase MoEForecaster: ensamble de expertos con gating adaptativo.

Vive en un módulo (y no dentro del notebook 08) para que el modelo serializado
(models/production/moe_forecaster.pkl) se pueda cargar desde cualquier script o
notebook con `pickle.load`, siempre que `src/` esté en el PYTHONPATH.
"""

from datetime import datetime

import numpy as np
import pandas as pd


class MoEForecaster:
    """
    Mixture-of-Experts Forecaster para demanda eléctrica del SIN colombiano.

    Gating adaptativo:
    - Alta confianza (min_conf ≥ threshold): hard gating → experto más probable
    - Baja confianza (min_conf < threshold): soft gating → combinación ponderada

    Parámetros
    ----------
    experts : dict
        {expert_id: sklearn_model} con modelos entrenados
    classifier_day : sklearn estimator con predict_proba
    classifier_week : sklearn estimator con predict_proba
    label_encoder_day : LabelEncoder para type_day
    label_encoder_week : LabelEncoder para type_week
    feature_cols : list of str — columnas de features en orden
    clf_feature_cols : list of str — columnas para los clasificadores
    gating_threshold : float — umbral de confianza mínima
    tau : float — temperatura para soft gating basado en distancia
    min_prob_soft : float — prob mínima para incluir experto en soft gating
    version : str — versión del modelo (se reporta en el model card)
    """

    def __init__(self, experts, classifier_day, classifier_week,
                 label_encoder_day, label_encoder_week,
                 feature_cols, clf_feature_cols,
                 gating_threshold=0.75, tau=1.0, min_prob_soft=0.05,
                 version="1.0.0"):
        self.experts             = experts
        self.clf_day             = classifier_day
        self.clf_week            = classifier_week
        self.le_day              = label_encoder_day
        self.le_week             = label_encoder_week
        self.feature_cols        = feature_cols
        self.clf_feature_cols    = clf_feature_cols
        self.gating_threshold    = gating_threshold
        self.tau                 = tau
        self.min_prob_soft       = min_prob_soft
        self.version             = version
        self.created_at          = datetime.now().isoformat()

    def _get_expert_features(self, X):
        """X puede ser DataFrame o np.array"""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_cols].values
        return X

    def _get_clf_features(self, X):
        if isinstance(X, pd.DataFrame):
            available = [c for c in self.clf_feature_cols if c in X.columns]
            if len(available) < len(self.clf_feature_cols):
                return X[self.feature_cols].values  # fallback
            return X[available].values
        return X  # si ya es ndarray, usar directo

    def predict(self, X):
        """
        Generar predicciones con gating adaptativo.

        X puede ser:
        - pd.DataFrame con columnas nombradas
        - np.ndarray en el orden de self.feature_cols

        Returns: np.ndarray de predicciones (kWh)
        """
        X_exp = self._get_expert_features(X)
        X_clf_arr = self._get_clf_features(X)
        n = len(X_exp)
        predictions = np.zeros(n)

        # Predicciones de confianza
        probs_day  = self.clf_day.predict_proba(X_clf_arr)   # (n, k_day)
        probs_week = self.clf_week.predict_proba(X_clf_arr)  # (n, k_week)
        classes_day  = self.le_day.classes_
        classes_week = self.le_week.classes_
        max_day  = probs_day.max(axis=1)
        max_week = probs_week.max(axis=1)
        min_conf = np.minimum(max_day, max_week)

        for i in range(n):
            x_i = X_exp[[i]]

            if min_conf[i] >= self.gating_threshold:
                # Hard gating: usar el experto más probable
                cd = classes_day[np.argmax(probs_day[i])]
                cw = classes_week[np.argmax(probs_week[i])]
                eid = f'{cd}_{cw}'
                if eid in self.experts:
                    predictions[i] = self.experts[eid].predict(x_i)[0]
                else:
                    # Fallback: experto más similar disponible
                    predictions[i] = self._fallback_predict(x_i, probs_day[i], probs_week[i],
                                                            classes_day, classes_week)
            else:
                # Soft gating: combinación ponderada
                predictions[i] = self._soft_predict(x_i, probs_day[i], probs_week[i],
                                                    classes_day, classes_week)

        return predictions

    def predict_with_metadata(self, X):
        """Predice y retorna metadatos de gating para monitoreo."""
        X_exp = self._get_expert_features(X)
        X_clf_arr = self._get_clf_features(X)
        n = len(X_exp)
        predictions = np.zeros(n)
        metadata = []

        probs_day  = self.clf_day.predict_proba(X_clf_arr)
        probs_week = self.clf_week.predict_proba(X_clf_arr)
        classes_day  = self.le_day.classes_
        classes_week = self.le_week.classes_
        max_day  = probs_day.max(axis=1)
        max_week = probs_week.max(axis=1)
        min_conf = np.minimum(max_day, max_week)

        for i in range(n):
            x_i = X_exp[[i]]
            cd = classes_day[np.argmax(probs_day[i])]
            cw = classes_week[np.argmax(probs_week[i])]
            eid = f'{cd}_{cw}'

            if min_conf[i] >= self.gating_threshold:
                mode = 'hard'
                if eid in self.experts:
                    predictions[i] = self.experts[eid].predict(x_i)[0]
                else:
                    predictions[i] = self._fallback_predict(x_i, probs_day[i], probs_week[i],
                                                            classes_day, classes_week)
                    mode = 'hard_fallback'
            else:
                mode = 'soft'
                predictions[i] = self._soft_predict(x_i, probs_day[i], probs_week[i],
                                                    classes_day, classes_week)

            metadata.append({
                'expert_id': eid,
                'conf_day': round(float(max_day[i]), 4),
                'conf_week': round(float(max_week[i]), 4),
                'min_conf': round(float(min_conf[i]), 4),
                'gating_mode': mode,
            })

        return predictions, pd.DataFrame(metadata)

    def _soft_predict(self, x_i, probs_d, probs_w, classes_d, classes_w):
        pred, w_total = 0.0, 0.0
        for di, cd in enumerate(classes_d):
            for wi, cw in enumerate(classes_w):
                p = probs_d[di] * probs_w[wi]
                if p > self.min_prob_soft:
                    eid = f'{cd}_{cw}'
                    if eid in self.experts:
                        pred += p * self.experts[eid].predict(x_i)[0]
                        w_total += p
        if w_total > 0:
            return pred / w_total
        return self._fallback_predict(x_i, probs_d, probs_w, classes_d, classes_w)

    def _fallback_predict(self, x_i, probs_d, probs_w, classes_d, classes_w):
        """Predicción de emergencia: promedio de todos los expertos disponibles."""
        if not self.experts:
            raise RuntimeError("MoEForecaster sin expertos: no se puede predecir.")
        return np.mean([m.predict(x_i)[0] for m in self.experts.values()])

    def get_model_card(self):
        return {
            'version': self.version,
            'created_at': self.created_at,
            'n_experts': len(self.experts),
            'expert_ids': list(self.experts.keys()),
            'gating_threshold': self.gating_threshold,
            'tau': self.tau,
            'feature_count': len(self.feature_cols),
            'clf_feature_count': len(self.clf_feature_cols),
        }
