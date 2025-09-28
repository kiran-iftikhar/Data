#Codes with respect to varibles:
#Codes for Capacitance:
# train_lstm_gru_bilstm_capacitance_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# --- load & columns ---
df = pd.read_excel('phy data.xlsx')
df.columns = [c.strip() for c in df.columns]
cap_col = 'Capacitance (µF)'
if cap_col not in df.columns: raise ValueError(f"Missing '{cap_col}'")
df = df.rename(columns={'Time (ms)':'Time', cap_col:'Capacitance'})[['Time','Capacitance']]

# --- cycles & split (80/20 each) ---
get_cycles = lambda d: (
    d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
    d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
    d[(d['Time']>=7.51)&(d['Time']<=10)].copy()
)
c1,c2,c3 = get_cycles(df)
split = lambda cdf: (cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te = split(c1); c2_tr,c2_te = split(c2); c3_tr,c3_te = split(c3)

# --- scaling (tanh range) ---
scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df  = pd.concat([c1_te,c2_te,c3_te],  ignore_index=True)
train_df['Capacitance_scaled'] = scaler.fit_transform(train_df[['Capacitance']])
test_df['Capacitance_scaled']  = scaler.transform(test_df[['Capacitance']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['Capacitance_scaled'] = scaler.transform(d[['Capacitance']])

# --- sequences ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series = train_df['Capacitance_scaled'].values
X_train, y_train = make_seq(series, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# --- builders with metrics ---
def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_lstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        LSTM(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_gru(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        GRU(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        GRU(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_bilstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=(look_back,1)), Dropout(0.2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

models = {'LSTM':build_lstm, 'GRU':build_gru, 'BiLSTM':build_bilstm}

# --- grid (exact) ---
units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None, 1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m = builder(look_back=look_back, units=u, lr=lr)
                h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                          callbacks=[es], verbose=0)
                mv = float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)] = mv
                training_histories[name][(u,lr,bs)] = h.history
                if mv < best[1]: best=((u,lr,bs), mv)
    (u,lr,bs), v = best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m = builder(look_back=look_back, units=u, lr=lr)
    h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
              callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)] = h.history

payload = {
    'models': tuned, 'best_units': best_units, 'best_lr': best_lr,
    'best_batch_size': best_batch, 'val_loss': best_val,
    'tuning_history': tuning_history, 'training_histories': training_histories,
    'scaler': scaler, 'look_back': look_back,
    'c1_test': c1_te, 'c2_test': c2_te, 'c3_test': c3_te,
    'cycle_names':['Cycle 1','Cycle 2','Cycle 3']
}
with open('results_lstm_gru_bilstm_capacitance_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_lstm_gru_bilstm_capacitance_tanh.pkl")
# train_cnn_lstm_gru_bilstm_capacitance_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
cap_col='Capacitance (µF)'
if cap_col not in df.columns: raise ValueError(f"Missing '{cap_col}'")
df=df.rename(columns={'Time (ms)':'Time', cap_col:'Capacitance'})[['Time','Capacitance']]

get_cycles=lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                     d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                     d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3=get_cycles(df)
split=lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)

scaler=MinMaxScaler(feature_range=(-1,1))
train_df=pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df =pd.concat([c1_te,c2_te,c3_te], ignore_index=True)
train_df['Capacitance_scaled']=scaler.fit_transform(train_df[['Capacitance']])
test_df['Capacitance_scaled']=scaler.transform(test_df[['Capacitance']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['Capacitance_scaled']=scaler.transform(d[['Capacitance']])

def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series=train_df['Capacitance_scaled'].values
X_train,y_train=make_seq(series,look_back)
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))

def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr), loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_cnn_lstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        LSTM(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_gru(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        GRU(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_bilstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

models={'CNN+LSTM':build_cnn_lstm,'CNN+GRU':build_cnn_gru,'CNN+BiLSTM':build_cnn_bilstm}

units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es=EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None,1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m=builder(look_back=look_back, units=u, lr=lr)
                h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                        callbacks=[es], verbose=0)
                mv=float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)]=mv
                training_histories[name][(u,lr,bs)]=h.history
                if mv<best[1]: best=((u,lr,bs),mv)
    (u,lr,bs),v=best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m=builder(look_back=look_back, units=u, lr=lr)
    h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
            callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)]=h.history

payload={'models':tuned,'best_units':best_units,'best_lr':best_lr,'best_batch_size':best_batch,
         'val_loss':best_val,'tuning_history':tuning_history,'training_histories':training_histories,
         'scaler':scaler,'look_back':look_back,'c1_test':c1_te,'c2_test':c2_te,'c3_test':c3_te,
         'cycle_names':['Cycle 1','Cycle 2','Cycle 3']}
with open('results_cnn_lstm_gru_bilstm_capacitance_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_cnn_lstm_gru_bilstm_capacitance_tanh.pkl")
# plot_best_model_predictions_capacitance.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error

# ---- Compact 3×3 in style; small one-line title, boxed two-column legend ----
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2,   # small, single-line heading
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,  # compact legend text
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})

def safe_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def make_seq(a, look_back=15):
    X, y = [], []
    for i in range(len(a) - look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

# Load results
d1 = safe_load('results_lstm_gru_bilstm_capacitance_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_capacitance_tanh.pkl')

models      = {**d1.get('models', {}), **d2.get('models', {})}
best_units  = {**d1.get('best_units', {}), **d2.get('best_units', {})}
best_lr     = {**d1.get('best_lr', {}), **d2.get('best_lr', {})}
best_batch  = {**d1.get('best_batch_size', {}), **d2.get('best_batch_size', {})}
scaler      = d1.get('scaler') or d2.get('scaler')
look_back   = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1', 'Cycle 2', 'Cycle 3']))
assert models and scaler is not None, "Models/scaler not found. Train scripts must be run first."

# Data
df = pd.read_excel('phy data.xlsx'); df.columns = [c.strip() for c in df.columns]
cap_col = 'Capacitance (µF)'
if cap_col not in df.columns: raise ValueError(f"Missing column '{cap_col}'.")
df = df.rename(columns={'Time (ms)':'Time', cap_col:'Capacitance'})[['Time','Capacitance']]

def get_cycles(d):
    c1 = d[(d['Time'] >= 0)    & (d['Time'] <= 2.515)].copy()
    c2 = d[(d['Time'] >= 2.52) & (d['Time'] <= 7.505)].copy()
    c3 = d[(d['Time'] >= 7.51) & (d['Time'] <= 10)].copy()
    return c1, c2, c3

def split_8020(cdf):
    k = int(0.8 * len(cdf)); return cdf.iloc[:k], cdf.iloc[k:]

c1, c2, c3 = get_cycles(df)
c1_tr, c1_te = split_8020(c1); c2_tr, c2_te = split_8020(c2); c3_tr, c3_te = split_8020(c3)
cycle_trains = [c1_tr, c2_tr, c3_tr]; cycle_tests  = [c1_te, c2_te, c3_te]

# Choose best model by avg Test RMSE
avg_test_rmse = {}
for mname, m in models.items():
    rmses = []
    for te in cycle_tests:
        te_scaled = te.copy()
        te_scaled['Capacitance_scaled'] = scaler.transform(te_scaled[['Capacitance']])
        arr = te_scaled['Capacitance_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte) == 0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        y_true = scaler.inverse_transform(yte.reshape(-1, 1))
        y_pred = scaler.inverse_transform(models[mname].predict(Xte, verbose=0))
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    if rmses: avg_test_rmse[mname] = float(np.mean(rmses))
assert len(avg_test_rmse) > 0, "No test evaluation available to choose the best model."

best_model_name = min(avg_test_rmse, key=avg_test_rmse.get)
best_model = models[best_model_name]
print(f"Best model (Avg Test RMSE): {best_model_name}  -> {avg_test_rmse[best_model_name]:.6f}")

# ---------- 3.0 in × 3.0 in; single boxed legend (two columns), tight x–legend gap ----------
fig_width_in  = 3.0
fig_height_in = 3.0

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs  = GridSpec(nrows=3, ncols=1, height_ratios=[8.7, 0.25, 7.05], figure=fig)
ax  = fig.add_subplot(gs[0])
ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")

ax.set_title(f"Best model: {best_model_name} — Predictions across cycles", pad=1.1)

# Colors (no gray/black); thin lines
train_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
test_colors  = ["#1f77b4", "#e377c2", "#17becf"]
pred_colors  = ["#d62728", "#bcbd22", "#8c564b"]
train_lw = 1.0; test_lw  = 1.4; pred_lw  = 1.0

# Train lines
for (tr, name, clr) in zip(cycle_trains, cycle_names, train_colors):
    ax.plot(tr['Time'].values, tr['Capacitance'].values,
            color=clr, linewidth=train_lw, alpha=0.98, label=f'{name} Train')

# Test + Predictions
for (te, name, tclr, pclr) in zip(cycle_tests, cycle_names, test_colors, pred_colors):
    te_scaled = te.copy()
    te_scaled['Capacitance_scaled'] = scaler.transform(te_scaled[['Capacitance']])
    arr = te_scaled['Capacitance_scaled'].values
    Xte, yte = make_seq(arr, look_back)
    if len(Xte) == 0: continue
    Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
    y_true = scaler.inverse_transform(yte.reshape(-1, 1))
    t_plot = te_scaled['Time'].values[look_back:]
    y_pred = scaler.inverse_transform(best_model.predict(Xte, verbose=0))
    ax.plot(t_plot, y_true, color=tclr, linewidth=test_lw, label=f'{name} Test Actual')
    ax.plot(t_plot, y_pred, color=pclr, linestyle='--', linewidth=pred_lw, alpha=0.98,
            label=f'{best_model_name} Prediction (Cycle {name.split()[-1]})')

ax.set_xlabel('Time (ms)', labelpad=0.8)
ax.set_ylabel('Capacitance (µF)')
for s in ax.spines.values(): s.set_visible(True)

# Legend (create it ON the legend axis so it actually shows)
handles, labels = ax.get_legend_handles_labels()
order = [
    "Cycle 1 Train", "Cycle 1 Test Actual", f"{best_model_name} Prediction (Cycle 1)",
    "Cycle 2 Train", "Cycle 2 Test Actual", f"{best_model_name} Prediction (Cycle 2)",
    "Cycle 3 Train", "Cycle 3 Test Actual", f"{best_model_name} Prediction (Cycle 3)",
]
map_h = {l: h for h, l in zip(handles, labels)}
labels_ord  = [l for l in order if l in map_h]
handles_ord = [map_h[l] for l in labels_ord]

leg = ax_leg.legend(
    handles_ord, labels_ord,
    loc="center",
    ncol=2,                # two columns inside ONE frame
    frameon=True,
    borderpad=0.10,
    columnspacing=0.40,
    handlelength=0.70,
    handletextpad=0.22,
    labelspacing=0.18,
    mode="expand",
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_facecolor("white")

# Tidy layout
fig.subplots_adjust(left=0.16, right=0.995, top=0.90, bottom=0.16, hspace=0.06)

# Save
fig.savefig('fig_best_model_predictions_capacitance.pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('fig_best_model_predictions_capacitance.png', dpi=600, bbox_inches='tight', facecolor='white')

print("Saved:")
print(" - fig_best_model_predictions_capacitance.pdf")
print(" - fig_best_model_predictions_capacitance.png")
# make_capacitance_figures_and_tables.py
import numpy as np, pickle, matplotlib.pyplot as plt, matplotlib as mpl, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# --- Unified compact style (matches 3×3 accepted plot) ---
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2,      # small, single-line titles
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,     # compact legend
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "--",
})

# --- helpers ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

def fig_ax_with_legend_row(fig_w=3.0, fig_h=3.0, top_title=""):
    """Create a 3-row layout: plot / tiny spacer / legend row.
       Legend is now closer to the x-axis by shrinking spacer and legend rows."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    # ↓ tighter gap: smaller spacer (0.06) and slightly shorter legend row (5.7)
    gs  = GridSpec(nrows=3, ncols=1, height_ratios=[9.25, 0.06, 5.69], figure=fig)
    ax  = fig.add_subplot(gs[0])
    ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
    ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")
    if top_title:
        ax.set_title(top_title, pad=1.1)
    return fig, ax, ax_leg

def place_legend_bottom(ax_leg, handles, labels, ncol=2):
    # Center the entire legend box and keep entries compact (no stretching to corners)
    leg = ax_leg.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),   # center of the legend row axis
        ncol=ncol,
        frameon=True,
        borderpad=0.10,
        columnspacing=0.32,
        handlelength=0.70,
        handletextpad=0.22,
        labelspacing=0.18,
    )
    fr = leg.get_frame()
    fr.set_edgecolor("black"); fr.set_linewidth(0.6); fr.set_facecolor("white")
    return leg

# --- load results (merge both) ---
def safe_load(path):
    try:
        with open(path,'rb') as f: return pickle.load(f)
    except: return {}
d1 = safe_load('results_lstm_gru_bilstm_capacitance_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_capacitance_tanh.pkl')

models = {**d1.get('models',{}), **d2.get('models',{})}
best_units = {**d1.get('best_units',{}), **d2.get('best_units',{})}
best_lr    = {**d1.get('best_lr',{}), **d2.get('best_lr',{})}
best_batch = {**d1.get('best_batch_size',{}), **d2.get('best_batch_size',{})}
histories  = {**d1.get('training_histories',{}), **d2.get('training_histories',{})}
val_loss   = {**d1.get('val_loss',{}), **d2.get('val_loss',{})}
scaler = d1.get('scaler') or d2.get('scaler')
look_back = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1','Cycle 2','Cycle 3']))

# --- rebuild data & splits for plotting ---
df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
cap_col = 'Capacitance (µF)';  assert cap_col in df.columns, f"Missing '{cap_col}'"
df = df.rename(columns={'Time (ms)':'Time', cap_col:'Capacitance'})[['Time','Capacitance']]
get_cycles = lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                       d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                       d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3 = get_cycles(df)
split = lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)
cycle_trains=[c1_tr,c2_tr,c3_tr]; cycle_tests=[c1_te,c2_te,c3_te]

# ------------------ FIGURE 1: Train/Test split by cycle ------------------
fig1, ax1, ax1_leg = fig_ax_with_legend_row(top_title="Train/Test split by cycle — Capacitance")
colors_train = ["#2ca02c", "#ff7f0e", "#9467bd"]
colors_test  = ["#1f77b4", "#e377c2", "#17becf"]
for (tr,te,name,ct,cv) in zip(cycle_trains, cycle_tests, cycle_names, colors_train, colors_test):
    ax1.plot(tr['Time'], tr['Capacitance'], color=ct, linewidth=1.2, label=f"{name} Train")
    ax1.plot(te['Time'], te['Capacitance'], color=cv, linestyle='--', linewidth=1.2, label=f"{name} Test")
ax1.set_xlabel("Time (ms)", labelpad=0.5)   # ↓ closer to legend
ax1.set_ylabel("Capacitance (µF)")
h,l = ax1.get_legend_handles_labels()
place_legend_bottom(ax1_leg, h, l, ncol=2)
# ↓ less bottom margin and tighter hspace, bringing legend closer
fig1.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 2: Combined predictions across cycles ------------------
def eval_and_plot_combined():
    fig2, ax2, ax2_leg = fig_ax_with_legend_row(top_title="Predictions across cycles — Combined view")
    model_colors = ["#d62728","#bcbd22","#8c564b","#17becf","#e377c2","#2ca02c"]
    test_actual_colors = ["#1f77b4","#e377c2","#17becf"]
    train_actual_colors= ["#2ca02c","#ff7f0e","#9467bd"]
    all_results = {m:{'rmse':[], 'mae':[], 'mse':[]} for m in models.keys()}

    # Train parts
    for (tr,name,clr) in zip(cycle_trains, cycle_names, train_actual_colors):
        ax2.plot(tr['Time'].values, tr['Capacitance'].values, color=clr, linewidth=1.1, alpha=0.95, label=f"{name} Train")

    # Test + predictions per model
    for ci,(te,name,tclr) in enumerate(zip(cycle_tests, cycle_names, test_actual_colors)):
        te_scaled = te.copy()
        te_scaled['Capacitance_scaled'] = scaler.transform(te_scaled[['Capacitance']])
        arr = te_scaled['Capacitance_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte)==0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        yte_inv = scaler.inverse_transform(yte.reshape(-1,1))
        t_plot = te_scaled['Time'].values[look_back:]
        ax2.plot(t_plot, yte_inv, color=tclr, linewidth=1.6, label=f"{name} Test Actual")

        for mi,(mname,m) in enumerate(models.items()):
            ypred = scaler.inverse_transform(m.predict(Xte, verbose=0))
            rmse = float(np.sqrt(mean_squared_error(yte_inv, ypred)))
            mae  = float(mean_absolute_error(yte_inv, ypred))
            mse  = float(mean_squared_error(yte_inv, ypred))
            all_results[mname]['rmse'].append(rmse)
            all_results[mname]['mae' ].append(mae)
            all_results[mname]['mse' ].append(mse)
            if ci==0:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95,
                         label=f"{mname} Prediction (Cycle {ci+1})")
            else:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95)

    ax2.set_xlabel('Time (ms)', labelpad=0.5)  # ↓ closer to legend
    ax2.set_ylabel('Capacitance (µF)')
    h,l = ax2.get_legend_handles_labels()
    place_legend_bottom(ax2_leg, h, l, ncol=2)
    fig2.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)
    return fig2, all_results

fig2, all_results = eval_and_plot_combined()

# ------------------ TABLES (train/val histories + test metrics) ------------------
def best_hist_for(model_name):
    params = (best_units.get(model_name,64), best_lr.get(model_name,0.001), best_batch.get(model_name,32))
    h = histories.get(model_name,{}).get(params, None)
    if h is None and histories.get(model_name,{}):
        h = list(histories[model_name].values())[0]
    return h or {}

train_rows=[]; val_rows=[]
for mname in models.keys():
    h = best_hist_for(mname)
    avg_train_mae = float(np.mean(h.get('mae', h.get('loss', [])) or [np.nan]))
    avg_train_rmse= float(np.mean(h.get('rmse', []) or [np.nan]))
    avg_val_mae   = float(np.mean(h.get('val_mae', h.get('val_loss', [])) or [np.nan]))
    avg_val_rmse  = float(np.mean(h.get('val_rmse', []) or [np.nan]))
    train_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_train_rmse:.6f}" if np.isfinite(avg_train_rmse) else 'nan',
                       'Avg MAE':f"{avg_train_mae:.6f}" if np.isfinite(avg_train_mae) else 'nan',
                       'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
    val_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_val_rmse:.6f}" if np.isfinite(avg_val_rmse) else 'nan',
                     'Avg MAE':f"{avg_val_mae:.6f}" if np.isfinite(avg_val_mae) else 'nan',
                     'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
train_df = pd.DataFrame(train_rows).sort_values('Model Name')
val_df   = pd.DataFrame(val_rows).sort_values('Model Name')

test_rows=[]
for mname,res in all_results.items():
    avg_rmse=float(np.mean(res['rmse'])) if res['rmse'] else np.nan
    avg_mae =float(np.mean(res['mae']))  if res['mae']  else np.nan
    test_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_rmse:.6f}" if np.isfinite(avg_rmse) else 'nan',
                      'Avg MAE':f"{avg_mae:.6f}"  if np.isfinite(avg_mae)  else 'nan',
                      'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),
                      'Best Batch':best_batch.get(mname,'N/A')})
test_df = pd.DataFrame(test_rows).sort_values('Model Name')

print("\n" + "="*60 + "\nTRAINING LOSS SUMMARY TABLE\n" + "="*60)
print(train_df.to_string(index=False))
print("\n" + "="*60 + "\nVALIDATION LOSS SUMMARY TABLE\n" + "="*60)
print(val_df.to_string(index=False))
print("\n" + "="*60 + "\nTEST LOSS SUMMARY TABLE\n" + "="*60)
print(test_df.to_string(index=False))

# ------------------ SUMMARY (Best Train/Val/Test) ------------------
def best_of(df, col):
    tmp = df[df[col] != 'nan'].copy()
    if tmp.empty: return ("N/A","nan")
    idx = tmp[col].astype(float).idxmin()
    r = tmp.loc[idx]; return (r['Model Name'], r[col])

best_train_mae  = best_of(train_df, 'Avg MAE')
best_train_rmse = best_of(train_df, 'Avg RMSE')
best_val_mae    = best_of(val_df,   'Avg MAE')
best_val_rmse   = best_of(val_df,   'Avg RMSE')

best_test_rmse_name = None; best_test_rmse_val = np.inf
best_test_mse_name  = None; best_test_mse_val  = np.inf
for mname,res in all_results.items():
    if res['rmse']:
        avg_rmse = float(np.mean(res['rmse']))
        if avg_rmse < best_test_rmse_val:
            best_test_rmse_val = avg_rmse; best_test_rmse_name = mname
    if res['mse']:
        avg_mse = float(np.mean(res['mse']))
        if avg_mse < best_test_mse_val:
            best_test_mse_val = avg_mse; best_test_mse_name = mname

print("\nSUMMARY")
print(f"Best Training Loss (MAE):  {best_train_mae[0]}  (Avg MAE: {best_train_mae[1]})")
print(f"Best Training Loss (RMSE): {best_train_rmse[0]} (Avg RMSE: {best_train_rmse[1]})")
print(f"Best Validation Loss (MAE):  {best_val_mae[0]}  (Avg MAE: {best_val_mae[1]})")
print(f"Best Validation Loss (RMSE): {best_val_rmse[0]} (Avg RMSE: {best_val_rmse[1]})")
print(f"Best Test Loss (RMSE): {best_test_rmse_name} (Avg RMSE: {best_test_rmse_val:.6f})")
print(f"Best Test Loss (MSE):  {best_test_mse_name} (Avg MSE:  {best_test_mse_val:.6f})")

# pick the best model by test RMSE for curves
best_model_name = best_test_rmse_name
def get_best_history_for_model(name):
    params=(best_units[name], best_lr[name], best_batch[name])
    return histories[name][params]
hbest = get_best_history_for_model(best_model_name)

# ------------------ FIGURE 3: Epoch vs RMSE (best model) ------------------
fig3, ax3, ax3_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs RMSE")
ax3.plot(hbest.get('rmse',[]), label='Train RMSE', color="#1f77b4", linewidth=1.1)
ax3.plot(hbest.get('val_rmse',[]), label='Val RMSE', linestyle='--', color="#d62728", linewidth=1.1)
ax3.set_xlabel('Epoch', labelpad=0.5)  # ↓ closer
ax3.set_ylabel('RMSE')
h,l = ax3.get_legend_handles_labels()
place_legend_bottom(ax3_leg, h, l, ncol=2)
fig3.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 4: Epoch vs MAE (best model) ------------------
fig4, ax4, ax4_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs MAE")
ax4.plot(hbest.get('mae', hbest.get('loss', [])), label='Train MAE', color="#2ca02c", linewidth=1.1)
ax4.plot(hbest.get('val_mae', hbest.get('val_loss', [])), label='Val MAE', linestyle='--', color="#ff7f0e", linewidth=1.1)
ax4.set_xlabel('Epoch', labelpad=0.5)  # ↓ closer
ax4.set_ylabel('MAE')
h,l = ax4.get_legend_handles_labels()
place_legend_bottom(ax4_leg, h, l, ncol=2)
fig4.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 5: Train vs Val (MAE) ------------------
fig5, ax5, ax5_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Train vs Validation (MAE)")
ax5.plot(hbest.get('loss',[]), label='Train (MAE)', color="#9467bd", linewidth=1.1)
ax5.plot(hbest.get('val_loss',[]), label='Val (MAE)', linestyle='--', color="#17becf", linewidth=1.1)
ax5.set_xlabel('Epoch', labelpad=0.5)  # ↓ closer
ax5.set_ylabel('MAE')
h,l = ax5.get_legend_handles_labels()
place_legend_bottom(ax5_leg, h, l, ncol=2)
fig5.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ SAVE ALL FIGURES TO PDFs ------------------
with PdfPages('capacitance_all_figures.pdf') as pdf:
    for fig in (fig1, fig2, fig3, fig4, fig5):
        pdf.savefig(fig, bbox_inches='tight')

fig1.savefig('fig_cap_train_test_split.pdf', bbox_inches='tight')
fig2.savefig('fig_cap_combined_predictions_with_train.pdf', bbox_inches='tight')
fig3.savefig('fig_cap_best_epoch_vs_rmse.pdf', bbox_inches='tight')
fig4.savefig('fig_cap_best_epoch_vs_mae.pdf', bbox_inches='tight')
fig5.savefig('fig_cap_best_train_vs_val_mae.pdf', bbox_inches='tight')

print("\nSaved PDFs:")
print("  - capacitance_all_figures.pdf")
print("  - fig_cap_train_test_split.pdf")
print("  - fig_cap_combined_predictions_with_train.pdf")
print("  - fig_cap_best_epoch_vs_rmse.pdf")
print("  - fig_cap_best_epoch_vs_mae.pdf")
print("  - fig_cap_best_train_vs_val_mae.pdf")
#################################################################################################
#Codes for Measured Polarization:
# train_lstm_gru_bilstm_polarization_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# --- load & columns ---
df = pd.read_excel('phy data.xlsx')
df.columns = [c.strip() for c in df.columns]
pol_col = 'Measured Polarization (µC/cm2)'
if pol_col not in df.columns: raise ValueError(f"Missing '{pol_col}'")
df = df.rename(columns={'Time (ms)':'Time', pol_col:'MeasuredPolarization'})[['Time','MeasuredPolarization']]

# --- cycles & split (80/20 each) ---
get_cycles = lambda d: (
    d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
    d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
    d[(d['Time']>=7.51)&(d['Time']<=10)].copy()
)
c1,c2,c3 = get_cycles(df)
split = lambda cdf: (cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te = split(c1); c2_tr,c2_te = split(c2); c3_tr,c3_te = split(c3)

# --- scaling (tanh range) ---
scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df  = pd.concat([c1_te,c2_te,c3_te],  ignore_index=True)
train_df['MeasuredPolarization_scaled'] = scaler.fit_transform(train_df[['MeasuredPolarization']])
test_df['MeasuredPolarization_scaled']  = scaler.transform(test_df[['MeasuredPolarization']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['MeasuredPolarization_scaled'] = scaler.transform(d[['MeasuredPolarization']])

# --- sequences ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series = train_df['MeasuredPolarization_scaled'].values
X_train, y_train = make_seq(series, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# --- builders with metrics ---
def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_lstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        LSTM(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_gru(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        GRU(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        GRU(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_bilstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=(look_back,1)), Dropout(0.2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

models = {'LSTM':build_lstm, 'GRU':build_gru, 'BiLSTM':build_bilstm}

# --- grid (exact) ---
units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None, 1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m = builder(look_back=look_back, units=u, lr=lr)
                h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                          callbacks=[es], verbose=0)
                mv = float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)] = mv
                training_histories[name][(u,lr,bs)] = h.history
                if mv < best[1]: best=((u,lr,bs), mv)
    (u,lr,bs), v = best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m = builder(look_back=look_back, units=u, lr=lr)
    h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
              callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)] = h.history

payload = {
    'models': tuned, 'best_units': best_units, 'best_lr': best_lr,
    'best_batch_size': best_batch, 'val_loss': best_val,
    'tuning_history': tuning_history, 'training_histories': training_histories,
    'scaler': scaler, 'look_back': look_back,
    'c1_test': c1_te, 'c2_test': c2_te, 'c3_test': c3_te,
    'cycle_names':['Cycle 1','Cycle 2','Cycle 3']
}
with open('results_lstm_gru_bilstm_polarization_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_lstm_gru_bilstm_polarization_tanh.pkl")
# train_cnn_lstm_gru_bilstm_polarization_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
pol_col='Measured Polarization (µC/cm2)'
if pol_col not in df.columns: raise ValueError(f"Missing '{pol_col}'")
df=df.rename(columns={'Time (ms)':'Time', pol_col:'MeasuredPolarization'})[['Time','MeasuredPolarization']]

get_cycles=lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                     d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                     d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3=get_cycles(df)
split=lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)

scaler=MinMaxScaler(feature_range=(-1,1))
train_df=pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df =pd.concat([c1_te,c2_te,c3_te], ignore_index=True)
train_df['MeasuredPolarization_scaled']=scaler.fit_transform(train_df[['MeasuredPolarization']])
test_df['MeasuredPolarization_scaled']=scaler.transform(test_df[['MeasuredPolarization']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['MeasuredPolarization_scaled']=scaler.transform(d[['MeasuredPolarization']])

def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series=train_df['MeasuredPolarization_scaled'].values
X_train,y_train=make_seq(series,look_back)
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))

def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr), loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_cnn_lstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        LSTM(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_gru(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        GRU(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_bilstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

models={'CNN+LSTM':build_cnn_lstm,'CNN+GRU':build_cnn_gru,'CNN+BiLSTM':build_cnn_bilstm}

units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es=EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None,1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m=builder(look_back=look_back, units=u, lr=lr)
                h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                        callbacks=[es], verbose=0)
                mv=float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)]=mv
                training_histories[name][(u,lr,bs)]=h.history
                if mv<best[1]: best=((u,lr,bs),mv)
    (u,lr,bs),v=best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m=builder(look_back=look_back, units=u, lr=lr)
    h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
            callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)]=h.history

payload={'models':tuned,'best_units':best_units,'best_lr':best_lr,'best_batch_size':best_batch,
         'val_loss':best_val,'tuning_history':tuning_history,'training_histories':training_histories,
         'scaler':scaler,'look_back':look_back,'c1_test':c1_te,'c2_test':c2_te,'c3_test':c3_te,
         'cycle_names':['Cycle 1','Cycle 2','Cycle 3']}
with open('results_cnn_lstm_gru_bilstm_polarization_tanh.pkl','wb') as f: pickle.dump(payload,f)
print("saved results_cnn_lstm_gru_bilstm_polarization_tanh.pkl")
# make_polarization_figures_and_tables.py
import numpy as np, pickle, matplotlib.pyplot as plt, matplotlib as mpl, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# --- Unified compact style (matches 3×3 accepted plot) ---
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "--",
})

# --- helpers ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

def fig_ax_with_legend_row(fig_w=3.0, fig_h=3.0, top_title=""):
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=3, ncols=1, height_ratios=[9.25, 0.06, 5.69], figure=fig)
    ax  = fig.add_subplot(gs[0])
    ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
    ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")
    if top_title:
        ax.set_title(top_title, pad=1.1)
    return fig, ax, ax_leg

def place_legend_bottom(ax_leg, handles, labels, ncol=2):
    leg = ax_leg.legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=ncol, frameon=True, borderpad=0.10, columnspacing=0.32,
        handlelength=0.70, handletextpad=0.22, labelspacing=0.18,
    )
    fr = leg.get_frame()
    fr.set_edgecolor("black"); fr.set_linewidth(0.6); fr.set_facecolor("white")
    return leg

# --- load results (merge both) ---
def safe_load(path):
    try:
        with open(path,'rb') as f: return pickle.load(f)
    except: return {}
d1 = safe_load('results_lstm_gru_bilstm_polarization_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_polarization_tanh.pkl')

models = {**d1.get('models',{}), **d2.get('models',{})}
best_units = {**d1.get('best_units',{}), **d2.get('best_units',{})}
best_lr    = {**d1.get('best_lr',{}), **d2.get('best_lr',{})}
best_batch = {**d1.get('best_batch_size',{}), **d2.get('best_batch_size',{})}
histories  = {**d1.get('training_histories',{}), **d2.get('training_histories',{})}
val_loss   = {**d1.get('val_loss',{}), **d2.get('val_loss',{})}
scaler = d1.get('scaler') or d2.get('scaler')
look_back = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1','Cycle 2','Cycle 3']))

# --- rebuild data & splits for plotting ---
df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
pol_col = 'Measured Polarization (µC/cm2)';  assert pol_col in df.columns, f"Missing '{pol_col}'"
df = df.rename(columns={'Time (ms)':'Time', pol_col:'MeasuredPolarization'})[['Time','MeasuredPolarization']]
get_cycles = lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                       d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                       d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3 = get_cycles(df)
split = lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)
cycle_trains=[c1_tr,c2_tr,c3_tr]; cycle_tests=[c1_te,c2_te,c3_te]

# ------------------ FIGURE 1: Train/Test split by cycle ------------------
fig1, ax1, ax1_leg = fig_ax_with_legend_row(top_title="Train/Test split by cycle — Measured Polarization")
colors_train = ["#2ca02c", "#ff7f0e", "#9467bd"]
colors_test  = ["#1f77b4", "#e377c2", "#17becf"]
for (tr,te,name,ct,cv) in zip(cycle_trains, cycle_tests, cycle_names, colors_train, colors_test):
    ax1.plot(tr['Time'], tr['MeasuredPolarization'], color=ct, linewidth=1.2, label=f"{name} Train")
    ax1.plot(te['Time'], te['MeasuredPolarization'], color=cv, linestyle='--', linewidth=1.2, label=f"{name} Test")
ax1.set_xlabel("Time (ms)", labelpad=0.5)
ax1.set_ylabel("Measured Polarization (µC/cm2)")
h,l = ax1.get_legend_handles_labels()
place_legend_bottom(ax1_leg, h, l, ncol=2)
fig1.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 2: Combined predictions across cycles ------------------
def eval_and_plot_combined():
    fig2, ax2, ax2_leg = fig_ax_with_legend_row(top_title="Predictions across cycles — Combined view")
    model_colors = ["#d62728","#bcbd22","#8c564b","#17becf","#e377c2","#2ca02c"]
    test_actual_colors = ["#1f77b4","#e377c2","#17becf"]
    train_actual_colors= ["#2ca02c","#ff7f0e","#9467bd"]
    all_results = {m:{'rmse':[], 'mae':[], 'mse':[]} for m in models.keys()}

    # Train parts
    for (tr,name,clr) in zip(cycle_trains, cycle_names, train_actual_colors):
        ax2.plot(tr['Time'].values, tr['MeasuredPolarization'].values, color=clr, linewidth=1.1, alpha=0.95, label=f"{name} Train")

    # Test + predictions per model
    for ci,(te,name,tclr) in enumerate(zip(cycle_tests, cycle_names, test_actual_colors)):
        te_scaled = te.copy()
        te_scaled['MeasuredPolarization_scaled'] = scaler.transform(te_scaled[['MeasuredPolarization']])
        arr = te_scaled['MeasuredPolarization_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte)==0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        yte_inv = scaler.inverse_transform(yte.reshape(-1,1))
        t_plot = te_scaled['Time'].values[look_back:]
        ax2.plot(t_plot, yte_inv, color=tclr, linewidth=1.6, label=f"{name} Test Actual")

        for mi,(mname,m) in enumerate(models.items()):
            ypred = scaler.inverse_transform(m.predict(Xte, verbose=0))
            rmse = float(np.sqrt(mean_squared_error(yte_inv, ypred)))
            mae  = float(mean_absolute_error(yte_inv, ypred))
            mse  = float(mean_squared_error(yte_inv, ypred))
            all_results[mname]['rmse'].append(rmse)
            all_results[mname]['mae' ].append(mae)
            all_results[mname]['mse' ].append(mse)
            if ci==0:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95,
                         label=f"{mname} Prediction (Cycle {ci+1})")
            else:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95)

    ax2.set_xlabel('Time (ms)', labelpad=0.5)
    ax2.set_ylabel('Measured Polarization (µC/cm2)')
    h,l = ax2.get_legend_handles_labels()
    place_legend_bottom(ax2_leg, h, l, ncol=2)
    fig2.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)
    return fig2, all_results

fig2, all_results = eval_and_plot_combined()

# ------------------ TABLES (train/val histories + test metrics) ------------------
def best_hist_for(model_name):
    params = (best_units.get(model_name,64), best_lr.get(model_name,0.001), best_batch.get(model_name,32))
    h = histories.get(model_name,{}).get(params, None)
    if h is None and histories.get(model_name,{}):
        h = list(histories[model_name].values())[0]
    return h or {}

train_rows=[]; val_rows=[]
for mname in models.keys():
    h = best_hist_for(mname)
    avg_train_mae = float(np.mean(h.get('mae', h.get('loss', [])) or [np.nan]))
    avg_train_rmse= float(np.mean(h.get('rmse', []) or [np.nan]))
    avg_val_mae   = float(np.mean(h.get('val_mae', h.get('val_loss', [])) or [np.nan]))
    avg_val_rmse  = float(np.mean(h.get('val_rmse', []) or [np.nan]))
    train_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_train_rmse:.6f}" if np.isfinite(avg_train_rmse) else 'nan',
                       'Avg MAE':f"{avg_train_mae:.6f}" if np.isfinite(avg_train_mae) else 'nan',
                       'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
    val_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_val_rmse:.6f}" if np.isfinite(avg_val_rmse) else 'nan',
                     'Avg MAE':f"{avg_val_mae:.6f}" if np.isfinite(avg_val_mae) else 'nan',
                     'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
train_df = pd.DataFrame(train_rows).sort_values('Model Name')
val_df   = pd.DataFrame(val_rows).sort_values('Model Name')

test_rows=[]
for mname,res in all_results.items():
    avg_rmse=float(np.mean(res['rmse'])) if res['rmse'] else np.nan
    avg_mae =float(np.mean(res['mae']))  if res['mae']  else np.nan
    test_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_rmse:.6f}" if np.isfinite(avg_rmse) else 'nan',
                      'Avg MAE':f"{avg_mae:.6f}"  if np.isfinite(avg_mae)  else 'nan',
                      'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),
                      'Best Batch':best_batch.get(mname,'N/A')})
test_df = pd.DataFrame(test_rows).sort_values('Model Name')

print("\n" + "="*60 + "\nTRAINING LOSS SUMMARY TABLE\n" + "="*60)
print(train_df.to_string(index=False))
print("\n" + "="*60 + "\nVALIDATION LOSS SUMMARY TABLE\n" + "="*60)
print(val_df.to_string(index=False))
print("\n" + "="*60 + "\nTEST LOSS SUMMARY TABLE\n" + "="*60)
print(test_df.to_string(index=False))

# ------------------ SUMMARY (Best Train/Val/Test) ------------------
def best_of(df, col):
    tmp = df[df[col] != 'nan'].copy()
    if tmp.empty: return ("N/A","nan")
    idx = tmp[col].astype(float).idxmin()
    r = tmp.loc[idx]; return (r['Model Name'], r[col])

best_train_mae  = best_of(train_df, 'Avg MAE')
best_train_rmse = best_of(train_df, 'Avg RMSE')
best_val_mae    = best_of(val_df,   'Avg MAE')
best_val_rmse   = best_of(val_df,   'Avg RMSE')

best_test_rmse_name = None; best_test_rmse_val = np.inf
best_test_mse_name  = None; best_test_mse_val  = np.inf
for mname,res in all_results.items():
    if res['rmse']:
        avg_rmse = float(np.mean(res['rmse']))
        if avg_rmse < best_test_rmse_val:
            best_test_rmse_val = avg_rmse; best_test_rmse_name = mname
    if res['mse']:
        avg_mse = float(np.mean(res['mse']))
        if avg_mse < best_test_mse_val:
            best_test_mse_val = avg_mse; best_test_mse_name = mname

print("\nSUMMARY")
print(f"Best Training Loss (MAE):  {best_train_mae[0]}  (Avg MAE: {best_train_mae[1]})")
print(f"Best Training Loss (RMSE): {best_train_rmse[0]} (Avg RMSE: {best_train_rmse[1]})")
print(f"Best Validation Loss (MAE):  {best_val_mae[0]}  (Avg MAE: {best_val_mae[1]})")
print(f"Best Validation Loss (RMSE): {best_val_rmse[0]} (Avg RMSE: {best_val_rmse[1]})")
print(f"Best Test Loss (RMSE): {best_test_rmse_name} (Avg RMSE: {best_test_rmse_val:.6f})")
print(f"Best Test Loss (MSE):  {best_test_mse_name} (Avg MSE:  {best_test_mse_val:.6f})")

# pick the best model by test RMSE for curves
best_model_name = best_test_rmse_name
def get_best_history_for_model(name):
    params=(best_units[name], best_lr[name], best_batch[name])
    return histories[name][params]
hbest = get_best_history_for_model(best_model_name)

# ------------------ FIGURE 3: Epoch vs RMSE (best model) ------------------
fig3, ax3, ax3_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs RMSE")
ax3.plot(hbest.get('rmse',[]), label='Train RMSE', color="#1f77b4", linewidth=1.1)
ax3.plot(hbest.get('val_rmse',[]), label='Val RMSE', linestyle='--', color="#d62728", linewidth=1.1)
ax3.set_xlabel('Epoch', labelpad=0.5)
ax3.set_ylabel('RMSE')
h,l = ax3.get_legend_handles_labels()
place_legend_bottom(ax3_leg, h, l, ncol=2)
fig3.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 4: Epoch vs MAE (best model) ------------------
fig4, ax4, ax4_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs MAE")
ax4.plot(hbest.get('mae', hbest.get('loss', [])), label='Train MAE', color="#2ca02c", linewidth=1.1)
ax4.plot(hbest.get('val_mae', hbest.get('val_loss', [])), label='Val MAE', linestyle='--', color="#ff7f0e", linewidth=1.1)
ax4.set_xlabel('Epoch', labelpad=0.5)
ax4.set_ylabel('MAE')
h,l = ax4.get_legend_handles_labels()
place_legend_bottom(ax4_leg, h, l, ncol=2)
fig4.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 5: Train vs Val (MAE) ------------------
fig5, ax5, ax5_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Train vs Validation (MAE)")
ax5.plot(hbest.get('loss',[]), label='Train (MAE)', color="#9467bd", linewidth=1.1)
ax5.plot(hbest.get('val_loss',[]), label='Val (MAE)', linestyle='--', color="#17becf", linewidth=1.1)
ax5.set_xlabel('Epoch', labelpad=0.5)
ax5.set_ylabel('MAE')
h,l = ax5.get_legend_handles_labels()
place_legend_bottom(ax5_leg, h, l, ncol=2)
fig5.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ SAVE ALL FIGURES TO PDFs ------------------
with PdfPages('polarization_all_figures.pdf') as pdf:
    for fig in (fig1, fig2, fig3, fig4, fig5):
        pdf.savefig(fig, bbox_inches='tight')

fig1.savefig('fig_pol_train_test_split.pdf', bbox_inches='tight')
fig2.savefig('fig_pol_combined_predictions_with_train.pdf', bbox_inches='tight')
fig3.savefig('fig_pol_best_epoch_vs_rmse.pdf', bbox_inches='tight')
fig4.savefig('fig_pol_best_epoch_vs_mae.pdf', bbox_inches='tight')
fig5.savefig('fig_pol_best_train_vs_val_mae.pdf', bbox_inches='tight')

print("\nSaved PDFs:")
print("  - polarization_all_figures.pdf")
print("  - fig_pol_train_test_split.pdf")
print("  - fig_pol_combined_predictions_with_train.pdf")
print("  - fig_pol_best_epoch_vs_rmse.pdf")
print("  - fig_pol_best_epoch_vs_mae.pdf")
print("  - fig_pol_best_train_vs_val_mae.pdf")
# plot_best_model_predictions_polarization.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error

# ---- Compact 3×3 in style; small one-line title, boxed two-column legend ----
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})

def safe_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def make_seq(a, look_back=15):
    X, y = [], []
    for i in range(len(a) - look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

# Load results
d1 = safe_load('results_lstm_gru_bilstm_polarization_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_polarization_tanh.pkl')

models      = {**d1.get('models', {}), **d2.get('models', {})}
best_units  = {**d1.get('best_units', {}), **d2.get('best_units', {})}
best_lr     = {**d1.get('best_lr', {}), **d2.get('best_lr', {})}
best_batch  = {**d1.get('best_batch_size', {}), **d2.get('best_batch_size', {})}
scaler      = d1.get('scaler') or d2.get('scaler')
look_back   = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1', 'Cycle 2', 'Cycle 3']))
assert models and scaler is not None, "Models/scaler not found. Train scripts must be run first."

# Data
df = pd.read_excel('phy data.xlsx'); df.columns = [c.strip() for c in df.columns]
pol_col = 'Measured Polarization (µC/cm2)'
if pol_col not in df.columns: raise ValueError(f"Missing column '{pol_col}'.")
df = df.rename(columns={'Time (ms)':'Time', pol_col:'MeasuredPolarization'})[['Time','MeasuredPolarization']]

def get_cycles(d):
    c1 = d[(d['Time'] >= 0)    & (d['Time'] <= 2.515)].copy()
    c2 = d[(d['Time'] >= 2.52) & (d['Time'] <= 7.505)].copy()
    c3 = d[(d['Time'] >= 7.51) & (d['Time'] <= 10)].copy()
    return c1, c2, c3

def split_8020(cdf):
    k = int(0.8 * len(cdf)); return cdf.iloc[:k], cdf.iloc[k:]

c1, c2, c3 = get_cycles(df)
c1_tr, c1_te = split_8020(c1); c2_tr, c2_te = split_8020(c2); c3_tr, c3_te = split_8020(c3)
cycle_trains = [c1_tr, c2_tr, c3_tr]; cycle_tests  = [c1_te, c2_te, c3_te]

# Choose best model by avg Test RMSE
avg_test_rmse = {}
for mname, m in models.items():
    rmses = []
    for te in cycle_tests:
        te_scaled = te.copy()
        te_scaled['MeasuredPolarization_scaled'] = scaler.transform(te_scaled[['MeasuredPolarization']])
        arr = te_scaled['MeasuredPolarization_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte) == 0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        y_true = scaler.inverse_transform(yte.reshape(-1, 1))
        y_pred = scaler.inverse_transform(models[mname].predict(Xte, verbose=0))
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    if rmses: avg_test_rmse[mname] = float(np.mean(rmses))
assert len(avg_test_rmse) > 0, "No test evaluation available to choose the best model."

best_model_name = min(avg_test_rmse, key=avg_test_rmse.get)
best_model = models[best_model_name]
print(f"Best model (Avg Test RMSE): {best_model_name}  -> {avg_test_rmse[best_model_name]:.6f}")

# ---------- 3.0 in × 3.0 in; single boxed legend (two columns), tight x–legend gap ----------
fig_width_in  = 3.0
fig_height_in = 3.0

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs  = GridSpec(nrows=3, ncols=1, height_ratios=[8.7, 0.25, 7.05], figure=fig)
ax  = fig.add_subplot(gs[0])
ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")

ax.set_title(f"Best model: {best_model_name} — Predictions across cycles", pad=1.1)

# Colors (no gray/black); thin lines
train_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
test_colors  = ["#1f77b4", "#e377c2", "#17becf"]
pred_colors  = ["#d62728", "#bcbd22", "#8c564b"]
train_lw = 1.0; test_lw  = 1.4; pred_lw  = 1.0

# Train lines
for (tr, name, clr) in zip(cycle_trains, cycle_names, train_colors):
    ax.plot(tr['Time'].values, tr['MeasuredPolarization'].values,
            color=clr, linewidth=train_lw, alpha=0.98, label=f'{name} Train')

# Test + Predictions
for (te, name, tclr, pclr) in zip(cycle_tests, cycle_names, test_colors, pred_colors):
    te_scaled = te.copy()
    te_scaled['MeasuredPolarization_scaled'] = scaler.transform(te_scaled[['MeasuredPolarization']])
    arr = te_scaled['MeasuredPolarization_scaled'].values
    Xte, yte = make_seq(arr, look_back)
    if len(Xte) == 0: continue
    Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
    y_true = scaler.inverse_transform(yte.reshape(-1, 1))
    t_plot = te_scaled['Time'].values[look_back:]
    y_pred = scaler.inverse_transform(best_model.predict(Xte, verbose=0))
    ax.plot(t_plot, y_true, color=tclr, linewidth=test_lw, label=f'{name} Test Actual')
    ax.plot(t_plot, y_pred, color=pclr, linestyle='--', linewidth=pred_lw, alpha=0.98,
            label=f'{best_model_name} Prediction (Cycle {name.split()[-1]})')

ax.set_xlabel('Time (ms)', labelpad=0.8)
ax.set_ylabel('Measured Polarization (µC/cm2)')
for s in ax.spines.values(): s.set_visible(True)

# Legend
handles, labels = ax.get_legend_handles_labels()
order = [
    "Cycle 1 Train", "Cycle 1 Test Actual", f"{best_model_name} Prediction (Cycle 1)",
    "Cycle 2 Train", "Cycle 2 Test Actual", f"{best_model_name} Prediction (Cycle 2)",
    "Cycle 3 Train", "Cycle 3 Test Actual", f"{best_model_name} Prediction (Cycle 3)",
]
map_h = {l: h for h, l in zip(handles, labels)}
labels_ord  = [l for l in order if l in map_h]
handles_ord = [map_h[l] for l in labels_ord]

leg = ax_leg.legend(
    handles_ord, labels_ord,
    loc="center",
    ncol=2,
    frameon=True,
    borderpad=0.10,
    columnspacing=0.40,
    handlelength=0.70,
    handletextpad=0.22,
    labelspacing=0.18,
    mode="expand",
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_facecolor("white")

fig.subplots_adjust(left=0.16, right=0.995, top=0.90, bottom=0.16, hspace=0.06)

# Save
fig.savefig('fig_best_model_predictions_polarization.pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('fig_best_model_predictions_polarization.png', dpi=600, bbox_inches='tight', facecolor='white')

print("Saved:")
print(" - fig_best_model_predictions_polarization.pdf")
print(" - fig_best_model_predictions_polarization.png")
####################################################################################
#Codes for Dielctric Constant
# train_lstm_gru_bilstm_dielectric_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# --- load & columns ---
df = pd.read_excel('phy data.xlsx')
df.columns = [c.strip() for c in df.columns]
die_col = 'Dielectric Constant'
if die_col not in df.columns: raise ValueError(f"Missing '{die_col}'")
df = df.rename(columns={'Time (ms)':'Time', die_col:'DielectricConstant'})[['Time','DielectricConstant']]

# --- cycles & split (80/20 each) ---
get_cycles = lambda d: (
    d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
    d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
    d[(d['Time']>=7.51)&(d['Time']<=10)].copy()
)
c1,c2,c3 = get_cycles(df)
split = lambda cdf: (cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te = split(c1); c2_tr,c2_te = split(c2); c3_tr,c3_te = split(c3)

# --- scaling (tanh range) ---
scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df  = pd.concat([c1_te,c2_te,c3_te],  ignore_index=True)
train_df['DielectricConstant_scaled'] = scaler.fit_transform(train_df[['DielectricConstant']])
test_df['DielectricConstant_scaled']  = scaler.transform(test_df[['DielectricConstant']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['DielectricConstant_scaled'] = scaler.transform(d[['DielectricConstant']])

# --- sequences ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series = train_df['DielectricConstant_scaled'].values
X_train, y_train = make_seq(series, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# --- builders with metrics ---
def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_lstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        LSTM(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_gru(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        GRU(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        GRU(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_bilstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=(look_back,1)), Dropout(0.2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

models = {'LSTM':build_lstm, 'GRU':build_gru, 'BiLSTM':build_bilstm}

# --- grid (exact) ---
units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None, 1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m = builder(look_back=look_back, units=u, lr=lr)
                h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                          callbacks=[es], verbose=0)
                mv = float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)] = mv
                training_histories[name][(u,lr,bs)] = h.history
                if mv < best[1]: best=((u,lr,bs), mv)
    (u,lr,bs), v = best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m = builder(look_back=look_back, units=u, lr=lr)
    h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
              callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)] = h.history

payload = {
    'models': tuned, 'best_units': best_units, 'best_lr': best_lr,
    'best_batch_size': best_batch, 'val_loss': best_val,
    'tuning_history': tuning_history, 'training_histories': training_histories,
    'scaler': scaler, 'look_back': look_back,
    'c1_test': c1_te, 'c2_test': c2_te, 'c3_test': c3_te,
    'cycle_names':['Cycle 1','Cycle 2','Cycle 3']
}
with open('results_lstm_gru_bilstm_dielectric_tanh.pkl','wb') as f: pickle.dump(payload,f)
print("saved results_lstm_gru_bilstm_dielectric_tanh.pkl")
# train_cnn_lstm_gru_bilstm_dielectric_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
die_col='Dielectric Constant'
if die_col not in df.columns: raise ValueError(f"Missing '{die_col}'")
df=df.rename(columns={'Time (ms)':'Time', die_col:'DielectricConstant'})[['Time','DielectricConstant']]

get_cycles=lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                     d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                     d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3=get_cycles(df)
split=lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)

scaler=MinMaxScaler(feature_range=(-1,1))
train_df=pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df =pd.concat([c1_te,c2_te,c3_te], ignore_index=True)
train_df['DielectricConstant_scaled']=scaler.fit_transform(train_df[['DielectricConstant']])
test_df['DielectricConstant_scaled']=scaler.transform(test_df[['DielectricConstant']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['DielectricConstant_scaled']=scaler.transform(d[['DielectricConstant']])

def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series=train_df['DielectricConstant_scaled'].values
X_train,y_train=make_seq(series,look_back)
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))

def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr), loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_cnn_lstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        LSTM(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_gru(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        GRU(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_bilstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

models={'CNN+LSTM':build_cnn_lstm,'CNN+GRU':build_cnn_gru,'CNN+BiLSTM':build_cnn_bilstm}

units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es=EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None,1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m=builder(look_back=look_back, units=u, lr=lr)
                h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                        callbacks=[es], verbose=0)
                mv=float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)]=mv
                training_histories[name][(u,lr,bs)]=h.history
                if mv<best[1]: best=((u,lr,bs),mv)
    (u,lr,bs),v=best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m=builder(look_back=look_back, units=u, lr=lr)
    h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
            callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)]=h.history

payload={'models':tuned,'best_units':best_units,'best_lr':best_lr,'best_batch_size':best_batch,
         'val_loss':best_val,'tuning_history':tuning_history,'training_histories':training_histories,
         'scaler':scaler,'look_back':look_back,'c1_test':c1_te,'c2_test':c2_te,'c3_test':c3_te,
         'cycle_names':['Cycle 1','Cycle 2','Cycle 3']}
with open('results_cnn_lstm_gru_bilstm_dielectric_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_cnn_lstm_gru_bilstm_dielectric_tanh.pkl")
# make_dielectric_figures_and_tables.py
import numpy as np, pickle, matplotlib.pyplot as plt, matplotlib as mpl, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# --- Unified compact style ---
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "--",
})

# --- helpers ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

def fig_ax_with_legend_row(fig_w=3.0, fig_h=3.0, top_title=""):
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=3, ncols=1, height_ratios=[9.25, 0.06, 5.69], figure=fig)
    ax  = fig.add_subplot(gs[0])
    ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
    ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")
    if top_title:
        ax.set_title(top_title, pad=1.1)
    return fig, ax, ax_leg

def place_legend_bottom(ax_leg, handles, labels, ncol=2):
    leg = ax_leg.legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=ncol, frameon=True, borderpad=0.10, columnspacing=0.32,
        handlelength=0.70, handletextpad=0.22, labelspacing=0.18,
    )
    fr = leg.get_frame()
    fr.set_edgecolor("black"); fr.set_linewidth(0.6); fr.set_facecolor("white")
    return leg

# --- load results (merge both) ---
def safe_load(path):
    try:
        with open(path,'rb') as f: return pickle.load(f)
    except: return {}
d1 = safe_load('results_lstm_gru_bilstm_dielectric_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_dielectric_tanh.pkl')

models = {**d1.get('models',{}), **d2.get('models',{})}
best_units = {**d1.get('best_units',{}), **d2.get('best_units',{})}
best_lr    = {**d1.get('best_lr',{}), **d2.get('best_lr',{})}
best_batch = {**d1.get('best_batch_size',{}), **d2.get('best_batch_size',{})}
histories  = {**d1.get('training_histories',{}), **d2.get('training_histories',{})}
val_loss   = {**d1.get('val_loss',{}), **d2.get('val_loss',{})}
scaler = d1.get('scaler') or d2.get('scaler')
look_back = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1','Cycle 2','Cycle 3']))

# --- rebuild data & splits for plotting ---
df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
die_col = 'Dielectric Constant';  assert die_col in df.columns, f"Missing '{die_col}'"
df = df.rename(columns={'Time (ms)':'Time', die_col:'DielectricConstant'})[['Time','DielectricConstant']]
get_cycles = lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                       d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                       d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3 = get_cycles(df)
split = lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)
cycle_trains=[c1_tr,c2_tr,c3_tr]; cycle_tests=[c1_te,c2_te,c3_te]

# ------------------ FIGURE 1: Train/Test split by cycle ------------------
fig1, ax1, ax1_leg = fig_ax_with_legend_row(top_title="Train/Test split by cycle — Dielectric Constant")
colors_train = ["#2ca02c", "#ff7f0e", "#9467bd"]
colors_test  = ["#1f77b4", "#e377c2", "#17becf"]
for (tr,te,name,ct,cv) in zip(cycle_trains, cycle_tests, cycle_names, colors_train, colors_test):
    ax1.plot(tr['Time'], tr['DielectricConstant'], color=ct, linewidth=1.2, label=f"{name} Train")
    ax1.plot(te['Time'], te['DielectricConstant'], color=cv, linestyle='--', linewidth=1.2, label=f"{name} Test")
ax1.set_xlabel("Time (ms)", labelpad=0.5)
ax1.set_ylabel("Dielectric Constant")
h,l = ax1.get_legend_handles_labels()
place_legend_bottom(ax1_leg, h, l, ncol=2)
fig1.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 2: Combined predictions across cycles ------------------
def eval_and_plot_combined():
    fig2, ax2, ax2_leg = fig_ax_with_legend_row(top_title="Predictions across cycles — Combined view")
    model_colors = ["#d62728","#bcbd22","#8c564b","#17becf","#e377c2","#2ca02c"]
    test_actual_colors = ["#1f77b4","#e377c2","#17becf"]
    train_actual_colors= ["#2ca02c","#ff7f0e","#9467bd"]
    all_results = {m:{'rmse':[], 'mae':[], 'mse':[]} for m in models.keys()}

    # Train parts
    for (tr,name,clr) in zip(cycle_trains, cycle_names, train_actual_colors):
        ax2.plot(tr['Time'].values, tr['DielectricConstant'].values, color=clr, linewidth=1.1, alpha=0.95, label=f"{name} Train")

    # Test + predictions per model
    for ci,(te,name,tclr) in enumerate(zip(cycle_tests, cycle_names, test_actual_colors)):
        te_scaled = te.copy()
        te_scaled['DielectricConstant_scaled'] = scaler.transform(te_scaled[['DielectricConstant']])
        arr = te_scaled['DielectricConstant_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte)==0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        yte_inv = scaler.inverse_transform(yte.reshape(-1,1))
        t_plot = te_scaled['Time'].values[look_back:]
        ax2.plot(t_plot, yte_inv, color=tclr, linewidth=1.6, label=f"{name} Test Actual")

        for mi,(mname,m) in enumerate(models.items()):
            ypred = scaler.inverse_transform(m.predict(Xte, verbose=0))
            rmse = float(np.sqrt(mean_squared_error(yte_inv, ypred)))
            mae  = float(mean_absolute_error(yte_inv, ypred))
            mse  = float(mean_squared_error(yte_inv, ypred))
            all_results[mname]['rmse'].append(rmse)
            all_results[mname]['mae' ].append(mae)
            all_results[mname]['mse' ].append(mse)
            if ci==0:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95,
                         label=f"{mname} Prediction (Cycle {ci+1})")
            else:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95)

    ax2.set_xlabel('Time (ms)', labelpad=0.5)
    ax2.set_ylabel('Dielectric Constant')
    h,l = ax2.get_legend_handles_labels()
    place_legend_bottom(ax2_leg, h, l, ncol=2)
    fig2.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)
    return fig2, all_results

fig2, all_results = eval_and_plot_combined()

# ------------------ TABLES (train/val histories + test metrics) ------------------
def best_hist_for(model_name):
    params = (best_units.get(model_name,64), best_lr.get(model_name,0.001), best_batch.get(model_name,32))
    h = histories.get(model_name,{}).get(params, None)
    if h is None and histories.get(model_name,{}):
        h = list(histories[model_name].values())[0]
    return h or {}

train_rows=[]; val_rows=[]
for mname in models.keys():
    h = best_hist_for(mname)
    avg_train_mae = float(np.mean(h.get('mae', h.get('loss', [])) or [np.nan]))
    avg_train_rmse= float(np.mean(h.get('rmse', []) or [np.nan]))
    avg_val_mae   = float(np.mean(h.get('val_mae', h.get('val_loss', [])) or [np.nan]))
    avg_val_rmse  = float(np.mean(h.get('val_rmse', []) or [np.nan]))
    train_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_train_rmse:.6f}" if np.isfinite(avg_train_rmse) else 'nan',
                       'Avg MAE':f"{avg_train_mae:.6f}" if np.isfinite(avg_train_mae) else 'nan',
                       'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
    val_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_val_rmse:.6f}" if np.isfinite(avg_val_rmse) else 'nan',
                     'Avg MAE':f"{avg_val_mae:.6f}" if np.isfinite(avg_val_mae) else 'nan',
                     'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
train_df = pd.DataFrame(train_rows).sort_values('Model Name')
val_df   = pd.DataFrame(val_rows).sort_values('Model Name')

test_rows=[]
for mname,res in all_results.items():
    avg_rmse=float(np.mean(res['rmse'])) if res['rmse'] else np.nan
    avg_mae =float(np.mean(res['mae']))  if res['mae']  else np.nan
    test_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_rmse:.6f}" if np.isfinite(avg_rmse) else 'nan',
                      'Avg MAE':f"{avg_mae:.6f}"  if np.isfinite(avg_mae)  else 'nan',
                      'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),
                      'Best Batch':best_batch.get(mname,'N/A')})
test_df = pd.DataFrame(test_rows).sort_values('Model Name')

print("\n" + "="*60 + "\nTRAINING LOSS SUMMARY TABLE\n" + "="*60)
print(train_df.to_string(index=False))
print("\n" + "="*60 + "\nVALIDATION LOSS SUMMARY TABLE\n" + "="*60)
print(val_df.to_string(index=False))
print("\n" + "="*60 + "\nTEST LOSS SUMMARY TABLE\n" + "="*60)
print(test_df.to_string(index=False))

# ------------------ SUMMARY (Best Train/Val/Test) ------------------
def best_of(df, col):
    tmp = df[df[col] != 'nan'].copy()
    if tmp.empty: return ("N/A","nan")
    idx = tmp[col].astype(float).idxmin()
    r = tmp.loc[idx]; return (r['Model Name'], r[col])

best_train_mae  = best_of(train_df, 'Avg MAE')
best_train_rmse = best_of(train_df, 'Avg RMSE')
best_val_mae    = best_of(val_df,   'Avg MAE')
best_val_rmse   = best_of(val_df,   'Avg RMSE')

best_test_rmse_name = None; best_test_rmse_val = np.inf
best_test_mse_name  = None; best_test_mse_val  = np.inf
for mname,res in all_results.items():
    if res['rmse']:
        avg_rmse = float(np.mean(res['rmse']))
        if avg_rmse < best_test_rmse_val:
            best_test_rmse_val = avg_rmse; best_test_rmse_name = mname
    if res['mse']:
        avg_mse = float(np.mean(res['mse']))
        if avg_mse < best_test_mse_val:
            best_test_mse_val = avg_mse; best_test_mse_name = mname

print("\nSUMMARY")
print(f"Best Training Loss (MAE):  {best_train_mae[0]}  (Avg MAE: {best_train_mae[1]})")
print(f"Best Training Loss (RMSE): {best_train_rmse[0]} (Avg RMSE: {best_train_rmse[1]})")
print(f"Best Validation Loss (MAE):  {best_val_mae[0]}  (Avg MAE: {best_val_mae[1]})")
print(f"Best Validation Loss (RMSE): {best_val_rmse[0]} (Avg RMSE: {best_val_rmse[1]})")
print(f"Best Test Loss (RMSE): {best_test_rmse_name} (Avg RMSE: {best_test_rmse_val:.6f})")
print(f"Best Test Loss (MSE):  {best_test_mse_name} (Avg MSE:  {best_test_mse_val:.6f})")

# pick the best model by test RMSE for curves
best_model_name = best_test_rmse_name
def get_best_history_for_model(name):
    params=(best_units[name], best_lr[name], best_batch[name])
    return histories[name][params]
hbest = get_best_history_for_model(best_model_name)

# ------------------ FIGURE 3: Epoch vs RMSE (best model) ------------------
fig3, ax3, ax3_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs RMSE")
ax3.plot(hbest.get('rmse',[]), label='Train RMSE', color="#1f77b4", linewidth=1.1)
ax3.plot(hbest.get('val_rmse',[]), label='Val RMSE', linestyle='--', color="#d62728", linewidth=1.1)
ax3.set_xlabel('Epoch', labelpad=0.5)
ax3.set_ylabel('RMSE')
h,l = ax3.get_legend_handles_labels()
place_legend_bottom(ax3_leg, h, l, ncol=2)
fig3.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 4: Epoch vs MAE (best model) ------------------
fig4, ax4, ax4_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs MAE")
ax4.plot(hbest.get('mae', hbest.get('loss', [])), label='Train MAE', color="#2ca02c", linewidth=1.1)
ax4.plot(hbest.get('val_mae', hbest.get('val_loss', [])), label='Val MAE', linestyle='--', color="#ff7f0e", linewidth=1.1)
ax4.set_xlabel('Epoch', labelpad=0.5)
ax4.set_ylabel('MAE')
h,l = ax4.get_legend_handles_labels()
place_legend_bottom(ax4_leg, h, l, ncol=2)
fig4.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 5: Train vs Val (MAE) ------------------
fig5, ax5, ax5_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Train vs Validation (MAE)")
ax5.plot(hbest.get('loss',[]), label='Train (MAE)', color="#9467bd", linewidth=1.1)
ax5.plot(hbest.get('val_loss',[]), label='Val (MAE)', linestyle='--', color="#17becf", linewidth=1.1)
ax5.set_xlabel('Epoch', labelpad=0.5)
ax5.set_ylabel('MAE')
h,l = ax5.get_legend_handles_labels()
place_legend_bottom(ax5_leg, h, l, ncol=2)
fig5.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ SAVE ALL FIGURES TO PDFs ------------------
with PdfPages('dielectric_all_figures.pdf') as pdf:
    for fig in (fig1, fig2, fig3, fig4, fig5):
        pdf.savefig(fig, bbox_inches='tight')

fig1.savefig('fig_die_train_test_split.pdf', bbox_inches='tight')
fig2.savefig('fig_die_combined_predictions_with_train.pdf', bbox_inches='tight')
fig3.savefig('fig_die_best_epoch_vs_rmse.pdf', bbox_inches='tight')
fig4.savefig('fig_die_best_epoch_vs_mae.pdf', bbox_inches='tight')
fig5.savefig('fig_die_best_train_vs_val_mae.pdf', bbox_inches='tight')

print("\nSaved PDFs:")
print("  - dielectric_all_figures.pdf")
print("  - fig_die_train_test_split.pdf")
print("  - fig_die_combined_predictions_with_train.pdf")
print("  - fig_die_best_epoch_vs_rmse.pdf")
print("  - fig_die_best_epoch_vs_mae.pdf")
print("  - fig_die_best_train_vs_val_mae.pdf")
# plot_best_model_predictions_dielectric.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error

# ---- Compact 3×3 in style; small one-line title, boxed two-column legend ----
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})

def safe_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def make_seq(a, look_back=15):
    X, y = [], []
    for i in range(len(a) - look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

# Load results
d1 = safe_load('results_lstm_gru_bilstm_dielectric_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_dielectric_tanh.pkl')

models      = {**d1.get('models', {}), **d2.get('models', {})}
best_units  = {**d1.get('best_units', {}), **d2.get('best_units', {})}
best_lr     = {**d1.get('best_lr', {}), **d2.get('best_lr', {})}
best_batch  = {**d1.get('best_batch_size', {}), **d2.get('best_batch_size', {})}
scaler      = d1.get('scaler') or d2.get('scaler')
look_back   = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1', 'Cycle 2', 'Cycle 3']))
assert models and scaler is not None, "Models/scaler not found. Train scripts must be run first."

# Data
df = pd.read_excel('phy data.xlsx'); df.columns = [c.strip() for c in df.columns]
die_col = 'Dielectric Constant'
if die_col not in df.columns: raise ValueError(f"Missing column '{die_col}'.")
df = df.rename(columns={'Time (ms)':'Time', die_col:'DielectricConstant'})[['Time','DielectricConstant']]

def get_cycles(d):
    c1 = d[(d['Time'] >= 0)    & (d['Time'] <= 2.515)].copy()
    c2 = d[(d['Time'] >= 2.52) & (d['Time'] <= 7.505)].copy()
    c3 = d[(d['Time'] >= 7.51) & (d['Time'] <= 10)].copy()
    return c1, c2, c3

def split_8020(cdf):
    k = int(0.8 * len(cdf)); return cdf.iloc[:k], cdf.iloc[k:]

c1, c2, c3 = get_cycles(df)
c1_tr, c1_te = split_8020(c1); c2_tr, c2_te = split_8020(c2); c3_tr, c3_te = split_8020(c3)
cycle_trains = [c1_tr, c2_tr, c3_tr]; cycle_tests  = [c1_te, c2_te, c3_te]

# Choose best model by avg Test RMSE
avg_test_rmse = {}
for mname, m in models.items():
    rmses = []
    for te in cycle_tests:
        te_scaled = te.copy()
        te_scaled['DielectricConstant_scaled'] = scaler.transform(te_scaled[['DielectricConstant']])
        arr = te_scaled['DielectricConstant_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte) == 0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        y_true = scaler.inverse_transform(yte.reshape(-1, 1))
        y_pred = scaler.inverse_transform(models[mname].predict(Xte, verbose=0))
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    if rmses: avg_test_rmse[mname] = float(np.mean(rmses))
assert len(avg_test_rmse) > 0, "No test evaluation available to choose the best model."

best_model_name = min(avg_test_rmse, key=avg_test_rmse.get)
best_model = models[best_model_name]
print(f"Best model (Avg Test RMSE): {best_model_name}  -> {avg_test_rmse[best_model_name]:.6f}")

# ---------- 3.0 in × 3.0 in; single boxed legend (two columns) ----------
fig_width_in  = 3.0
fig_height_in = 3.0

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs  = GridSpec(nrows=3, ncols=1, height_ratios=[8.7, 0.25, 7.05], figure=fig)
ax  = fig.add_subplot(gs[0])
ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")

ax.set_title(f"Best model: {best_model_name} — Predictions across cycles", pad=1.1)

# Colors
train_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
test_colors  = ["#1f77b4", "#e377c2", "#17becf"]
pred_colors  = ["#d62728", "#bcbd22", "#8c564b"]
train_lw = 1.0; test_lw  = 1.4; pred_lw  = 1.0

# Train lines
for (tr, name, clr) in zip(cycle_trains, cycle_names, train_colors):
    ax.plot(tr['Time'].values, tr['DielectricConstant'].values,
            color=clr, linewidth=train_lw, alpha=0.98, label=f'{name} Train')

# Test + Predictions
for (te, name, tclr, pclr) in zip(cycle_tests, cycle_names, test_colors, pred_colors):
    te_scaled = te.copy()
    te_scaled['DielectricConstant_scaled'] = scaler.transform(te_scaled[['DielectricConstant']])
    arr = te_scaled['DielectricConstant_scaled'].values
    Xte, yte = make_seq(arr, look_back)
    if len(Xte) == 0: continue
    Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
    y_true = scaler.inverse_transform(yte.reshape(-1, 1))
    t_plot = te_scaled['Time'].values[look_back:]
    y_pred = scaler.inverse_transform(best_model.predict(Xte, verbose=0))
    ax.plot(t_plot, y_true, color=tclr, linewidth=test_lw, label=f'{name} Test Actual')
    ax.plot(t_plot, y_pred, color=pclr, linestyle='--', linewidth=pred_lw, alpha=0.98,
            label=f'{best_model_name} Prediction (Cycle {name.split()[-1]})')

ax.set_xlabel('Time (ms)', labelpad=0.8)
ax.set_ylabel('Dielectric Constant')
for s in ax.spines.values(): s.set_visible(True)

# Legend (boxed, two columns)
handles, labels = ax.get_legend_handles_labels()
order = [
    "Cycle 1 Train", "Cycle 1 Test Actual", f"{best_model_name} Prediction (Cycle 1)",
    "Cycle 2 Train", "Cycle 2 Test Actual", f"{best_model_name} Prediction (Cycle 2)",
    "Cycle 3 Train", "Cycle 3 Test Actual", f"{best_model_name} Prediction (Cycle 3)",
]
map_h = {l: h for h, l in zip(handles, labels)}
labels_ord  = [l for l in order if l in map_h]
handles_ord = [map_h[l] for l in labels_ord]

leg = ax_leg.legend(
    handles_ord, labels_ord,
    loc="center",
    ncol=2,
    frameon=True,
    borderpad=0.10,
    columnspacing=0.40,
    handlelength=0.70,
    handletextpad=0.22,
    labelspacing=0.18,
    mode="expand",
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_facecolor("white")

fig.subplots_adjust(left=0.16, right=0.995, top=0.90, bottom=0.16, hspace=0.06)

# Save
fig.savefig('fig_best_model_predictions_dielectric.pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('fig_best_model_predictions_dielectric.png', dpi=600, bbox_inches='tight', facecolor='white')

print("Saved:")
print(" - fig_best_model_predictions_dielectric.pdf")
print(" - fig_best_model_predictions_dielectric.png")
#################################################################################################
#Codes for Instantaneous Current
# train_lstm_gru_bilstm_current_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

# --- load & columns ---
df = pd.read_excel('phy data.xlsx')
df.columns = [c.strip() for c in df.columns]
cur_col = 'Instantaneous Current (mA)'
if cur_col not in df.columns: raise ValueError(f"Missing '{cur_col}'")
df = df.rename(columns={'Time (ms)':'Time', cur_col:'InstantaneousCurrent'})[['Time','InstantaneousCurrent']]

# --- cycles & split (80/20 each) ---
get_cycles = lambda d: (
    d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
    d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
    d[(d['Time']>=7.51)&(d['Time']<=10)].copy()
)
c1,c2,c3 = get_cycles(df)
split = lambda cdf: (cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te = split(c1); c2_tr,c2_te = split(c2); c3_tr,c3_te = split(c3)

# --- scaling (tanh range) ---
scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df  = pd.concat([c1_te,c2_te,c3_te],  ignore_index=True)
train_df['InstantaneousCurrent_scaled'] = scaler.fit_transform(train_df[['InstantaneousCurrent']])
test_df['InstantaneousCurrent_scaled']  = scaler.transform(test_df[['InstantaneousCurrent']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['InstantaneousCurrent_scaled'] = scaler.transform(d[['InstantaneousCurrent']])

# --- sequences ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series = train_df['InstantaneousCurrent_scaled'].values
X_train, y_train = make_seq(series, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# --- builders with metrics ---
def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr),
              loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_lstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        LSTM(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_gru(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        GRU(units, return_sequences=True, input_shape=(look_back,1)), Dropout(0.2),
        GRU(units), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

def build_bilstm(look_back=15, units=64, lr=1e-3):
    m = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=(look_back,1)), Dropout(0.2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='tanh')
    ]); return compile_model(m, lr)

models = {'LSTM':build_lstm, 'GRU':build_gru, 'BiLSTM':build_bilstm}

# --- grid (exact) ---
units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None, 1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m = builder(look_back=look_back, units=u, lr=lr)
                h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                          callbacks=[es], verbose=0)
                mv = float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)] = mv
                training_histories[name][(u,lr,bs)] = h.history
                if mv < best[1]: best=((u,lr,bs), mv)
    (u,lr,bs), v = best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m = builder(look_back=look_back, units=u, lr=lr)
    h = m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
              callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)] = h.history

payload = {
    'models': tuned, 'best_units': best_units, 'best_lr': best_lr,
    'best_batch_size': best_batch, 'val_loss': best_val,
    'tuning_history': tuning_history, 'training_histories': training_histories,
    'scaler': scaler, 'look_back': look_back,
    'c1_test': c1_te, 'c2_test': c2_te, 'c3_test': c3_te,
    'cycle_names':['Cycle 1','Cycle 2','Cycle 3']
}
with open('results_lstm_gru_bilstm_current_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_lstm_gru_bilstm_current_tanh.pkl")
# train_cnn_lstm_gru_bilstm_current_tanh.py
import numpy as np, pandas as pd, tensorflow as tf, random, pickle, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

SEED=42
np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED)

df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
cur_col='Instantaneous Current (mA)'
if cur_col not in df.columns: raise ValueError(f"Missing '{cur_col}'")
df=df.rename(columns={'Time (ms)':'Time', cur_col:'InstantaneousCurrent'})[['Time','InstantaneousCurrent']]

get_cycles=lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                     d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                     d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3=get_cycles(df)
split=lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)

scaler=MinMaxScaler(feature_range=(-1,1))
train_df=pd.concat([c1_tr,c2_tr,c3_tr], ignore_index=True)
test_df =pd.concat([c1_te,c2_te,c3_te], ignore_index=True)
train_df['InstantaneousCurrent_scaled']=scaler.fit_transform(train_df[['InstantaneousCurrent']])
test_df['InstantaneousCurrent_scaled']=scaler.transform(test_df[['InstantaneousCurrent']])
for d in (c1_tr,c1_te,c2_tr,c2_te,c3_tr,c3_te):
    d['InstantaneousCurrent_scaled']=scaler.transform(d[['InstantaneousCurrent']])

def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

look_back=15
series=train_df['InstantaneousCurrent_scaled'].values
X_train,y_train=make_seq(series,look_back)
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))

def compile_model(m, lr):
    m.compile(optimizer=Adam(learning_rate=lr), loss='mae',
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                       tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return m

def build_cnn_lstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        LSTM(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_gru(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        GRU(units), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

def build_cnn_bilstm(look_back=15, units=64, lr=1e-3):
    m=Sequential([
        Conv1D(filters=units,kernel_size=3,activation='relu',input_shape=(look_back,1)),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units)), Dropout(0.2),
        Dense(16,activation='relu'), Dense(1,activation='tanh')
    ]); return compile_model(m, lr)

models={'CNN+LSTM':build_cnn_lstm,'CNN+GRU':build_cnn_gru,'CNN+BiLSTM':build_cnn_bilstm}

units_list=[64,128]; lr_list=[0.001,0.01]; batch_list=[32,64]; epochs=50
tuning_history={k:{} for k in models}; training_histories={k:{} for k in models}
best_units={}; best_lr={}; best_batch={}; best_val={}; tuned={}
es=EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for name,builder in models.items():
    best=(None,1e9)
    for u in units_list:
        for lr in lr_list:
            for bs in batch_list:
                m=builder(look_back=look_back, units=u, lr=lr)
                h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
                        callbacks=[es], verbose=0)
                mv=float(np.min(h.history['val_loss']))
                tuning_history[name][(u,lr,bs)]=mv
                training_histories[name][(u,lr,bs)]=h.history
                if mv<best[1]: best=((u,lr,bs),mv)
    (u,lr,bs),v=best
    best_units[name]=u; best_lr[name]=lr; best_batch[name]=bs; best_val[name]=v
    m=builder(look_back=look_back, units=u, lr=lr)
    h=m.fit(X_train,y_train,epochs=epochs,batch_size=bs,validation_split=0.2,
            callbacks=[es], verbose=0)
    tuned[name]=m
    training_histories[name][(u,lr,bs)]=h.history

payload={'models':tuned,'best_units':best_units,'best_lr':best_lr,'best_batch_size':best_batch,
         'val_loss':best_val,'tuning_history':tuning_history,'training_histories':training_histories,
         'scaler':scaler,'look_back':look_back,'c1_test':c1_te,'c2_test':c2_te,'c3_test':c3_te,
         'cycle_names':['Cycle 1','Cycle 2','Cycle 3']}
with open('results_cnn_lstm_gru_bilstm_current_tanh.pkl','wb') as f: pickle.dump(payload,f)
print(" saved results_cnn_lstm_gru_bilstm_current_tanh.pkl")
# make_current_figures_and_tables.py
import numpy as np, pickle, matplotlib.pyplot as plt, matplotlib as mpl, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# --- compact style ---
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2, "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "--",
})

# --- helpers ---
def make_seq(a, look_back=15):
    X,y=[],[]
    for i in range(len(a)-look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
        # end
    return np.array(X), np.array(y)

def fig_ax_with_legend_row(fig_w=3.0, fig_h=3.0, top_title=""):
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=3, ncols=1, height_ratios=[9.25, 0.06, 5.69], figure=fig)
    ax  = fig.add_subplot(gs[0])
    ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
    ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")
    if top_title:
        ax.set_title(top_title, pad=1.1)
    return fig, ax, ax_leg

def place_legend_bottom(ax_leg, handles, labels, ncol=2):
    leg = ax_leg.legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5),
        ncol=ncol, frameon=True, borderpad=0.10, columnspacing=0.32,
        handlelength=0.70, handletextpad=0.22, labelspacing=0.18,
    )
    fr = leg.get_frame()
    fr.set_edgecolor("black"); fr.set_linewidth(0.6); fr.set_facecolor("white")
    return leg

# --- load results (merge both) ---
def safe_load(path):
    try:
        with open(path,'rb') as f: return pickle.load(f)
    except: return {}
d1 = safe_load('results_lstm_gru_bilstm_current_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_current_tanh.pkl')

models = {**d1.get('models',{}), **d2.get('models',{})}
best_units = {**d1.get('best_units',{}), **d2.get('best_units',{})}
best_lr    = {**d1.get('best_lr',{}), **d2.get('best_lr',{})}
best_batch = {**d1.get('best_batch_size',{}), **d2.get('best_batch_size',{})}
histories  = {**d1.get('training_histories',{}), **d2.get('training_histories',{})}
val_loss   = {**d1.get('val_loss',{}), **d2.get('val_loss',{})}
scaler = d1.get('scaler') or d2.get('scaler')
look_back = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1','Cycle 2','Cycle 3']))

# --- rebuild data & splits for plotting ---
df = pd.read_excel('phy data.xlsx'); df.columns=[c.strip() for c in df.columns]
cur_col = 'Instantaneous Current (mA)';  assert cur_col in df.columns, f"Missing '{cur_col}'"
df = df.rename(columns={'Time (ms)':'Time', cur_col:'InstantaneousCurrent'})[['Time','InstantaneousCurrent']]
get_cycles = lambda d:(d[(d['Time']>=0)&(d['Time']<=2.515)].copy(),
                       d[(d['Time']>=2.52)&(d['Time']<=7.505)].copy(),
                       d[(d['Time']>=7.51)&(d['Time']<=10)].copy())
c1,c2,c3 = get_cycles(df)
split = lambda cdf:(cdf.iloc[:int(0.8*len(cdf))], cdf.iloc[int(0.8*len(cdf)):])
c1_tr,c1_te=split(c1); c2_tr,c2_te=split(c2); c3_tr,c3_te=split(c3)
cycle_trains=[c1_tr,c2_tr,c3_tr]; cycle_tests=[c1_te,c2_te,c3_te]

# ------------------ FIGURE 1: Train/Test split by cycle ------------------
fig1, ax1, ax1_leg = fig_ax_with_legend_row(top_title="Train/Test split by cycle — Instantaneous Current")
colors_train = ["#2ca02c", "#ff7f0e", "#9467bd"]
colors_test  = ["#1f77b4", "#e377c2", "#17becf"]
for (tr,te,name,ct,cv) in zip(cycle_trains, cycle_tests, cycle_names, colors_train, colors_test):
    ax1.plot(tr['Time'], tr['InstantaneousCurrent'], color=ct, linewidth=1.2, label=f"{name} Train")
    ax1.plot(te['Time'], te['InstantaneousCurrent'], color=cv, linestyle='--', linewidth=1.2, label=f"{name} Test")
ax1.set_xlabel("Time (ms)", labelpad=0.5)
ax1.set_ylabel("Instantaneous Current (mA)")
h,l = ax1.get_legend_handles_labels()
place_legend_bottom(ax1_leg, h, l, ncol=2)
fig1.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 2: Combined predictions across cycles ------------------
def eval_and_plot_combined():
    fig2, ax2, ax2_leg = fig_ax_with_legend_row(top_title="Predictions across cycles — Combined view")
    model_colors = ["#d62728","#bcbd22","#8c564b","#17becf","#e377c2","#2ca02c"]
    test_actual_colors = ["#1f77b4","#e377c2","#17becf"]
    train_actual_colors= ["#2ca02c","#ff7f0e","#9467bd"]
    all_results = {m:{'rmse':[], 'mae':[], 'mse':[]} for m in models.keys()}

    # Train parts
    for (tr,name,clr) in zip(cycle_trains, cycle_names, train_actual_colors):
        ax2.plot(tr['Time'].values, tr['InstantaneousCurrent'].values, color=clr, linewidth=1.1, alpha=0.95, label=f"{name} Train")

    # Test + predictions per model
    for ci,(te,name,tclr) in enumerate(zip(cycle_tests, cycle_names, test_actual_colors)):
        te_scaled = te.copy()
        te_scaled['InstantaneousCurrent_scaled'] = scaler.transform(te_scaled[['InstantaneousCurrent']])
        arr = te_scaled['InstantaneousCurrent_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte)==0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        yte_inv = scaler.inverse_transform(yte.reshape(-1,1))
        t_plot = te_scaled['Time'].values[look_back:]
        ax2.plot(t_plot, yte_inv, color=tclr, linewidth=1.6, label=f"{name} Test Actual")

        for mi,(mname,m) in enumerate(models.items()):
            ypred = scaler.inverse_transform(m.predict(Xte, verbose=0))
            rmse = float(np.sqrt(mean_squared_error(yte_inv, ypred)))
            mae  = float(mean_absolute_error(yte_inv, ypred))
            mse  = float(mean_squared_error(yte_inv, ypred))
            all_results[mname]['rmse'].append(rmse)
            all_results[mname]['mae' ].append(mae)
            all_results[mname]['mse' ].append(mse)
            if ci==0:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95,
                         label=f"{mname} Prediction (Cycle {ci+1})")
            else:
                ax2.plot(t_plot, ypred, linestyle='--', linewidth=1.1,
                         color=model_colors[mi % len(model_colors)], alpha=0.95)

    ax2.set_xlabel('Time (ms)', labelpad=0.5)
    ax2.set_ylabel('Instantaneous Current (mA)')
    h,l = ax2.get_legend_handles_labels()
    place_legend_bottom(ax2_leg, h, l, ncol=2)
    fig2.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)
    return fig2, all_results

fig2, all_results = eval_and_plot_combined()

# ------------------ TABLES (train/val histories + test metrics) ------------------
def best_hist_for(model_name):
    params = (best_units.get(model_name,64), best_lr.get(model_name,0.001), best_batch.get(model_name,32))
    h = histories.get(model_name,{}).get(params, None)
    if h is None and histories.get(model_name,{}):
        h = list(histories[model_name].values())[0]
    return h or {}

train_rows=[]; val_rows=[]
for mname in models.keys():
    h = best_hist_for(mname)
    avg_train_mae = float(np.mean(h.get('mae', h.get('loss', [])) or [np.nan]))
    avg_train_rmse= float(np.mean(h.get('rmse', []) or [np.nan]))
    avg_val_mae   = float(np.mean(h.get('val_mae', h.get('val_loss', [])) or [np.nan]))
    avg_val_rmse  = float(np.mean(h.get('val_rmse', []) or [np.nan]))
    train_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_train_rmse:.6f}" if np.isfinite(avg_train_rmse) else 'nan',
                       'Avg MAE':f"{avg_train_mae:.6f}" if np.isfinite(avg_train_mae) else 'nan',
                       'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
    val_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_val_rmse:.6f}" if np.isfinite(avg_val_rmse) else 'nan',
                     'Avg MAE':f"{avg_val_mae:.6f}" if np.isfinite(avg_val_mae) else 'nan',
                     'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),'Best Batch':best_batch.get(mname,'N/A')})
train_df = pd.DataFrame(train_rows).sort_values('Model Name')
val_df   = pd.DataFrame(val_rows).sort_values('Model Name')

test_rows=[]
for mname,res in all_results.items():
    avg_rmse=float(np.mean(res['rmse'])) if res['rmse'] else np.nan
    avg_mae =float(np.mean(res['mae']))  if res['mae']  else np.nan
    test_rows.append({'Model Name':mname,'Avg RMSE':f"{avg_rmse:.6f}" if np.isfinite(avg_rmse) else 'nan',
                      'Avg MAE':f"{avg_mae:.6f}"  if np.isfinite(avg_mae)  else 'nan',
                      'Best Units':best_units.get(mname,'N/A'),'Best LR':best_lr.get(mname,'N/A'),
                      'Best Batch':best_batch.get(mname,'N/A')})
test_df = pd.DataFrame(test_rows).sort_values('Model Name')

print("\n" + "="*60 + "\nTRAINING LOSS SUMMARY TABLE\n" + "="*60)
print(train_df.to_string(index=False))
print("\n" + "="*60 + "\nVALIDATION LOSS SUMMARY TABLE\n" + "="*60)
print(val_df.to_string(index=False))
print("\n" + "="*60 + "\nTEST LOSS SUMMARY TABLE\n" + "="*60)
print(test_df.to_string(index=False))

# ------------------ SUMMARY (Best Train/Val/Test) ------------------
def best_of(df, col):
    tmp = df[df[col] != 'nan'].copy()
    if tmp.empty: return ("N/A","nan")
    idx = tmp[col].astype(float).idxmin()
    r = tmp.loc[idx]; return (r['Model Name'], r[col])

best_train_mae  = best_of(train_df, 'Avg MAE')
best_train_rmse = best_of(train_df, 'Avg RMSE')
best_val_mae    = best_of(val_df,   'Avg MAE')
best_val_rmse   = best_of(val_df,   'Avg RMSE')

best_test_rmse_name = None; best_test_rmse_val = np.inf
best_test_mse_name  = None; best_test_mse_val  = np.inf
for mname,res in all_results.items():
    if res['rmse']:
        avg_rmse = float(np.mean(res['rmse']))
        if avg_rmse < best_test_rmse_val:
            best_test_rmse_val = avg_rmse; best_test_rmse_name = mname
    if res['mse']:
        avg_mse = float(np.mean(res['mse']))
        if avg_mse < best_test_mse_val:
            best_test_mse_val = avg_mse; best_test_mse_name = mname

print("\nSUMMARY")
print(f"Best Training Loss (MAE):  {best_train_mae[0]}  (Avg MAE: {best_train_mae[1]})")
print(f"Best Training Loss (RMSE): {best_train_rmse[0]} (Avg RMSE: {best_train_rmse[1]})")
print(f"Best Validation Loss (MAE):  {best_val_mae[0]}  (Avg MAE: {best_val_mae[1]})")
print(f"Best Validation Loss (RMSE): {best_val_rmse[0]} (Avg RMSE: {best_val_rmse[1]})")
print(f"Best Test Loss (RMSE): {best_test_rmse_name} (Avg RMSE: {best_test_rmse_val:.6f})")
print(f"Best Test Loss (MSE):  {best_test_mse_name} (Avg MSE:  {best_test_mse_val:.6f})")

# pick the best model by test RMSE for curves
best_model_name = best_test_rmse_name
def get_best_history_for_model(name):
    params=(best_units[name], best_lr[name], best_batch[name])
    return histories[name][params]
hbest = get_best_history_for_model(best_model_name)

# ------------------ FIGURE 3: Epoch vs RMSE (best model) ------------------
fig3, ax3, ax3_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs RMSE")
ax3.plot(hbest.get('rmse',[]), label='Train RMSE', color="#1f77b4", linewidth=1.1)
ax3.plot(hbest.get('val_rmse',[]), label='Val RMSE', linestyle='--', color="#d62728", linewidth=1.1)
ax3.set_xlabel('Epoch', labelpad=0.5)
ax3.set_ylabel('RMSE')
h,l = ax3.get_legend_handles_labels()
place_legend_bottom(ax3_leg, h, l, ncol=2)
fig3.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 4: Epoch vs MAE (best model) ------------------
fig4, ax4, ax4_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Epoch vs MAE")
ax4.plot(hbest.get('mae', hbest.get('loss', [])), label='Train MAE', color="#2ca02c", linewidth=1.1)
ax4.plot(hbest.get('val_mae', hbest.get('val_loss', [])), label='Val MAE', linestyle='--', color="#ff7f0e", linewidth=1.1)
ax4.set_xlabel('Epoch', labelpad=0.5)
ax4.set_ylabel('MAE')
h,l = ax4.get_legend_handles_labels()
place_legend_bottom(ax4_leg, h, l, ncol=2)
fig4.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ FIGURE 5: Train vs Val (MAE) ------------------
fig5, ax5, ax5_leg = fig_ax_with_legend_row(top_title=f"{best_model_name}: Train vs Validation (MAE)")
ax5.plot(hbest.get('loss',[]), label='Train (MAE)', color="#9467bd", linewidth=1.1)
ax5.plot(hbest.get('val_loss',[]), label='Val (MAE)', linestyle='--', color="#17becf", linewidth=1.1)
ax5.set_xlabel('Epoch', labelpad=0.5)
ax5.set_ylabel('MAE')
h,l = ax5.get_legend_handles_labels()
place_legend_bottom(ax5_leg, h, l, ncol=2)
fig5.subplots_adjust(left=0.16,right=0.995,top=0.90,bottom=0.13,hspace=0.02)

# ------------------ SAVE ALL FIGURES TO PDFs ------------------
with PdfPages('current_all_figures.pdf') as pdf:
    for fig in (fig1, fig2, fig3, fig4, fig5):
        pdf.savefig(fig, bbox_inches='tight')

fig1.savefig('fig_cur_train_test_split.pdf', bbox_inches='tight')
fig2.savefig('fig_cur_combined_predictions_with_train.pdf', bbox_inches='tight')
fig3.savefig('fig_cur_best_epoch_vs_rmse.pdf', bbox_inches='tight')
fig4.savefig('fig_cur_best_epoch_vs_mae.pdf', bbox_inches='tight')
fig5.savefig('fig_cur_best_train_vs_val_mae.pdf', bbox_inches='tight')

print("\nSaved PDFs:")
print("  - current_all_figures.pdf")
print("  - fig_cur_train_test_split.pdf")
print("  - fig_cur_combined_predictions_with_train.pdf")
print("  - fig_cur_best_epoch_vs_rmse.pdf")
print("  - fig_cur_best_epoch_vs_mae.pdf")
print("  - fig_cur_best_train_vs_val_mae.pdf")
# plot_best_model_predictions_current.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error

# ---- Compact 3×3 in style; small one-line title, boxed two-column legend ----
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "axes.titlesize": 6.2,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.fontsize": 4.7,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
})

def safe_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def make_seq(a, look_back=15):
    X, y = [], []
    for i in range(len(a) - look_back):
        X.append(a[i:i+look_back]); y.append(a[i+look_back])
    return np.array(X), np.array(y)

# Load results
d1 = safe_load('results_lstm_gru_bilstm_current_tanh.pkl')
d2 = safe_load('results_cnn_lstm_gru_bilstm_current_tanh.pkl')

models      = {**d1.get('models', {}), **d2.get('models', {})}
best_units  = {**d1.get('best_units', {}), **d2.get('best_units', {})}
best_lr     = {**d1.get('best_lr', {}), **d2.get('best_lr', {})}
best_batch  = {**d1.get('best_batch_size', {}), **d2.get('best_batch_size', {})}
scaler      = d1.get('scaler') or d2.get('scaler')
look_back   = d1.get('look_back', d2.get('look_back', 15))
cycle_names = d1.get('cycle_names', d2.get('cycle_names', ['Cycle 1', 'Cycle 2', 'Cycle 3']))
assert models and scaler is not None, "Models/scaler not found. Train scripts must be run first."

# Data
df = pd.read_excel('phy data.xlsx'); df.columns = [c.strip() for c in df.columns]
cur_col = 'Instantaneous Current (mA)'
if cur_col not in df.columns: raise ValueError(f"Missing column '{cur_col}'.")
df = df.rename(columns={'Time (ms)':'Time', cur_col:'InstantaneousCurrent'})[['Time','InstantaneousCurrent']]

def get_cycles(d):
    c1 = d[(d['Time'] >= 0)    & (d['Time'] <= 2.515)].copy()
    c2 = d[(d['Time'] >= 2.52) & (d['Time'] <= 7.505)].copy()
    c3 = d[(d['Time'] >= 7.51) & (d['Time'] <= 10)].copy()
    return c1, c2, c3

def split_8020(cdf):
    k = int(0.8 * len(cdf)); return cdf.iloc[:k], cdf.iloc[k:]

c1, c2, c3 = get_cycles(df)
c1_tr, c1_te = split_8020(c1); c2_tr, c2_te = split_8020(c2); c3_tr, c3_te = split_8020(c3)
cycle_trains = [c1_tr, c2_tr, c3_tr]; cycle_tests  = [c1_te, c2_te, c3_te]

# Choose best model by avg Test RMSE
avg_test_rmse = {}
for mname, m in models.items():
    rmses = []
    for te in cycle_tests:
        te_scaled = te.copy()
        te_scaled['InstantaneousCurrent_scaled'] = scaler.transform(te_scaled[['InstantaneousCurrent']])
        arr = te_scaled['InstantaneousCurrent_scaled'].values
        Xte, yte = make_seq(arr, look_back)
        if len(Xte) == 0: continue
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
        y_true = scaler.inverse_transform(yte.reshape(-1, 1))
        y_pred = scaler.inverse_transform(models[mname].predict(Xte, verbose=0))
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
    if rmses: avg_test_rmse[mname] = float(np.mean(rmses))
assert len(avg_test_rmse) > 0, "No test evaluation available to choose the best model."

best_model_name = min(avg_test_rmse, key=avg_test_rmse.get)
best_model = models[best_model_name]
print(f"Best model (Avg Test RMSE): {best_model_name}  -> {avg_test_rmse[best_model_name]:.6f}")

# ---------- 3.0 in × 3.0 in; single boxed legend (two columns) ----------
fig_width_in  = 3.0
fig_height_in = 3.0

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs  = GridSpec(nrows=3, ncols=1, height_ratios=[8.7, 0.25, 7.05], figure=fig)
ax  = fig.add_subplot(gs[0])
ax_space = fig.add_subplot(gs[1]); ax_space.axis("off")
ax_leg   = fig.add_subplot(gs[2]); ax_leg.axis("off")

ax.set_title(f"Best model: {best_model_name} — Predictions across cycles", pad=1.1)

# Colors
train_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
test_colors  = ["#1f77b4", "#e377c2", "#17becf"]
pred_colors  = ["#d62728", "#bcbd22", "#8c564b"]
train_lw = 1.0; test_lw  = 1.4; pred_lw  = 1.0

# Train lines
for (tr, name, clr) in zip(cycle_trains, cycle_names, train_colors):
    ax.plot(tr['Time'].values, tr['InstantaneousCurrent'].values,
            color=clr, linewidth=train_lw, alpha=0.98, label=f'{name} Train')

# Test + Predictions
for (te, name, tclr, pclr) in zip(cycle_tests, cycle_names, test_colors, pred_colors):
    te_scaled = te.copy()
    te_scaled['InstantaneousCurrent_scaled'] = scaler.transform(te_scaled[['InstantaneousCurrent']])
    arr = te_scaled['InstantaneousCurrent_scaled'].values
    Xte, yte = make_seq(arr, look_back)
    if len(Xte) == 0: continue
    Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
    y_true = scaler.inverse_transform(yte.reshape(-1, 1))
    t_plot = te_scaled['Time'].values[look_back:]
    y_pred = scaler.inverse_transform(best_model.predict(Xte, verbose=0))
    ax.plot(t_plot, y_true, color=tclr, linewidth=test_lw, label=f'{name} Test Actual')
    ax.plot(t_plot, y_pred, color=pclr, linestyle='--', linewidth=pred_lw, alpha=0.98,
            label=f'{best_model_name} Prediction (Cycle {name.split()[-1]})')

ax.set_xlabel('Time (ms)', labelpad=0.8)
ax.set_ylabel('Instantaneous Current (mA)')
for s in ax.spines.values(): s.set_visible(True)

# Legend (boxed, two columns)
handles, labels = ax.get_legend_handles_labels()
order = [
    "Cycle 1 Train", "Cycle 1 Test Actual", f"{best_model_name} Prediction (Cycle 1)",
    "Cycle 2 Train", "Cycle 2 Test Actual", f"{best_model_name} Prediction (Cycle 2)",
    "Cycle 3 Train", "Cycle 3 Test Actual", f"{best_model_name} Prediction (Cycle 3)",
]
map_h = {l: h for h, l in zip(handles, labels)}
labels_ord  = [l for l in order if l in map_h]
handles_ord = [map_h[l] for l in labels_ord]

leg = ax_leg.legend(
    handles_ord, labels_ord,
    loc="center",
    ncol=2,
    frameon=True,
    borderpad=0.10,
    columnspacing=0.40,
    handlelength=0.70,
    handletextpad=0.22,
    labelspacing=0.18,
    mode="expand",
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_facecolor("white")

fig.subplots_adjust(left=0.16, right=0.995, top=0.90, bottom=0.16, hspace=0.06)

# Save
fig.savefig('fig_best_model_predictions_current.pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('fig_best_model_predictions_current.png', dpi=600, bbox_inches='tight', facecolor='white')

print("Saved:")
print(" - fig_best_model_predictions_current.pdf")
print(" - fig_best_model_predictions_current.png")







