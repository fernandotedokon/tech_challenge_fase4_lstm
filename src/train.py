import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from src.data_loader import load_data
from src.preprocessing import create_sequences
from api.train_status import TRAIN_STATUS

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)



# ===============================
# CALLBACK DE PROGRESSO
# ===============================
class TrainStatusCallback(Callback):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        TRAIN_STATUS[self.symbol]["progress"] = int((epoch + 1) / TRAIN_STATUS[self.symbol]["epochs"] * 100)


# ===============================
# FUNÇÃO PARA SALVAR SCALER
# ===============================
def save_scaler(scaler, symbol: str):
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}.pkl")
    joblib.dump(scaler, scaler_path)


# ===============================
# FUNÇÃO DE AVALIAÇÃO
# ===============================
def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


# ===============================
# CONSTRUÇÃO DO MODELO
# ===============================
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model


# ===============================
# FUNÇÃO PRINCIPAL DE TREINO
# ===============================
def train_model(symbol: str, ticker: str, epochs: int = 20,
                start_date: str = "2018-01-01", end_date: str = "2024-12-31",
                window_size: int = 60):

    # Inicializa status
    TRAIN_STATUS[symbol] = {"status": "IN_PROGRESS", "progress": 0, "epochs": epochs}

    # 1️⃣ Carregar dados
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        TRAIN_STATUS[symbol]["status"] = "FAILED"
        raise ValueError(f"Nenhum dado disponível para {symbol} entre {start_date} e {end_date}")

    # 2️⃣ Pré-processamento
    X, y, scaler = create_sequences(df, window_size)

    # 3️⃣ Split treino/validação
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 4️⃣ Construir modelo
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # 5️⃣ Callback de progresso
    callback = TrainStatusCallback(symbol)

    # 6️⃣ Treino
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[callback],
        verbose=1
    )

    # 7️⃣ Avaliação
    preds = model.predict(X_val)
    metrics = evaluate(y_val, preds)

    # 8️⃣ Salvar modelo e scaler
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.keras")
    model.save(model_path)
    save_scaler(scaler, symbol)

    # Atualiza status final
    TRAIN_STATUS[symbol]["status"] = "COMPLETED"
    TRAIN_STATUS[symbol]["progress"] = 100
    TRAIN_STATUS[symbol]["metrics"] = metrics
    TRAIN_STATUS[symbol]["model_path"] = model_path

    return {
        "symbol": symbol,
        "epochs": epochs,
        "model_path": model_path,
        "metrics": metrics
    }

   

if __name__ == "__main__":
    results = []
    for name, ticker in SYMBOLS.items():
        results.append(train_symbol(name, ticker))

    print("Treinamento finalizado:")
    print(results)
