import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from double_ml import train_double_ml, predict_cate

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_dataset(returns_df, macro_df, treatment_series, window):
    ret_win = returns_df.iloc[-window:]
    macro_win = macro_df.iloc[-window:] if not macro_df.empty else pd.DataFrame(0, index=ret_win.index, columns=config.MACRO_COLUMNS)
    treat_win = treatment_series.iloc[-window:]
    common = ret_win.index.intersection(macro_win.index).intersection(treat_win.index)
    ret_win = ret_win.loc[common]
    macro_win = macro_win.loc[common]
    treat_win = treat_win.loc[common]
    # Features: macro levels + market return (if available)
    X = macro_win.values
    if 'SPY' in ret_win.columns:
        market_ret = ret_win['SPY'].values
    else:
        market_ret = ret_win.mean(axis=1).values
    X = np.column_stack([X, market_ret])
    T = treat_win.values
    Y = ret_win.shift(-1).dropna().values
    X = X[:-1]
    T = T[:-1]
    return X, T, Y

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    treatment_series = data_manager.get_treatment_series(df, config.TREATMENT_COL, window=252, threshold=config.TREATMENT_THRESHOLD)

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (DoubleML) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro = data_manager.get_macro_data(df)
        if macro.empty:
            print("  No macro data; using zeros")
            macro = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            X, T, Y = create_dataset(returns, macro, treatment_series, win)
            if X.shape[0] < 20:
                continue
            num_tasks = Y.shape[1]
            cates = np.zeros(num_tasks)
            for etf_idx, etf in enumerate(tickers):
                y = Y[:, etf_idx]
                model = train_double_ml(X, T, y,
                                        model_y=config.MODEL_Y,
                                        model_t=config.MODEL_T,
                                        cv=config.CROSSFOLD)
                # Predict CATE for the last feature vector
                test_X = X[-1:].reshape(1, -1)
                cate = predict_cate(model, test_X)
                cates[etf_idx] = cate[0]
            scores = {tickers[i]: cates[i] for i in range(num_tasks)}
            window_results[win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "cate": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by CATE: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/double_ml_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Double/Debiased ML Engine complete ===")

if __name__ == "__main__":
    main()
