import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import math
import collections
import time
import matplotlib.font_manager as fm

# 日本語フォントの設定（任意）
# 環境に合わせて適切なフォント名を指定してください。
# 例: Windowsでは "Meiryo UI", "Yu Gothic", macOSでは "Hiragino Sans" など
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Meiryo UI", "Yu Gothic", "Hiragino Sans", "Apple Color Emoji", "Segoe UI Emoji"]

# === 設定 ===
FS = 44100      # サンプリング周波数

def is_valid(value):
    """NaNではない有効な数値かを確認するヘルパー関数"""
    return value is not None and not math.isnan(value)

def get_formants(audio, sr=FS):
    """
    ParSelmouthを使用して音声データからF1, F2フォルマントを抽出する関数。
    """
    print("DEBUG: get_formants called.") # デバッグ: 関数呼び出しを確認

    # NumPy配列からParSelmouthのSoundオブジェクトを作成
    # データをint16として渡し、ParSelmouthが内部で適切に処理できるようにします。
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    
    # 音声の長さが極端に短い場合にエラーになる可能性があるのでチェック
    if snd.get_total_duration() == 0:
        print("DEBUG: Audio duration is zero.")
        return float('nan'), float('nan')

    try:
        # Formant objectを生成
        # フォルマント抽出パラメータを調整して、より安定した結果を得ることを試みます。
        # time_step: 分析する時間間隔 (秒)。短すぎると分析負荷が増え、長すぎると解像度が落ちる。
        # maximum_formant: 検出するフォルマントの最大周波数 (Hz)。話者の声質（男性/女性）で調整。
        #                  男性: ~5000Hz, 女性: ~5500Hzが目安。
        # window_length: フォルマント分析に使用する分析窓の長さ (秒)。
        #                短すぎると安定せずNaNが出やすい。
        formant = snd.to_formant_burg(
            time_step=0.01,
            maximum_formant=5500, # 例: 5500Hz (女性の声や高めの声に対応)
            max_number_of_formants=5,
            window_length=0.1 # 例: 30ms　0.03だったのを0.1に変更　■■■■■■■■■■■■■■■■■■■■■■■■■
        )
        
        # フォルマントの分析時間を取得（通常、チャンクの中央付近）
        # get_start_time() は Sound オブジェクトの開始時間。
        # formant.get_time_from_frame(1) などで最初のフレームの時間を取得できますが、
        # ここでは単純に最初のフレームの値を参照することを想定します。
        # parselmouthのFormantオブジェクトは内部的に時間情報を持っています。
        # 通常、get_value_at_time() は指定された時間に最も近い分析フレームの値を返します。
        
        # 簡易的に、最初の有効な時間でフォルマントを取得
        analysis_time = snd.get_total_duration() / 2 # チャンクの中央付近を分析点とする

        f1 = formant.get_value_at_time(1, analysis_time)
        f2 = formant.get_value_at_time(2, analysis_time)
        
        print(f"DEBUG: Formants extracted: F1={f1}, F2={f2}")
        
        return f1, f2
    except parselmouth.PraatError as praat_err:
        # Praatからの特定のエラーをキャッチ
        print(f"ERROR (PraatError) in get_formants: {praat_err}")
        # PraatErrorの一般的な原因: ピッチが検出できない、音量が小さすぎる、フォルマントが検出できないなど
        return float('nan'), float('nan')
    except Exception as e:
        # その他の予期せぬエラーをキャッチ
        print(f"ERROR in get_formants: {e}")
        return float('nan'), float('nan')

class FormantProcessor(AudioProcessorBase):
    def __init__(self):
        print("DEBUG: FormantProcessor __init__ called.")
        self.f1_history = collections.deque(maxlen=10) # F1履歴
        self.f2_history = collections.deque(maxlen=10) # F2履歴
        self.latest_f1 = None
        self.latest_f2 = None
        self.output_dict = {"f1": None, "f2": None}
        self.last_log_time = time.time() # デバッグログ用のタイムスタンプ
        
        # 音声チャンクを結合するためのバッファ
        self.audio_buffer = np.array([], dtype=np.int16) # 音声バッファ
        self.buffer_duration_sec = 0.1 # 0.1秒分の音声でフォルマントを分析
        self.buffer_size_samples = int(self.buffer_duration_sec * FS) # 必要なサンプル数

    def recv(self, frame):
        current_time = time.time()
        # 0.5秒ごとにデバッグメッセージを出力して、recvが呼ばれていることを確認
        if current_time - self.last_log_time > 0.5:
            print(f"DEBUG: recv method called at {current_time:.2f}")
            self.last_log_time = current_time

        try:
            # streamlit-webrtc からの frame は av.AudioFrame オブジェクト
            # .to_ndarray() で NumPy 配列に変換。
            # 古いPyAVバージョンだと format/layout 引数がないため削除。
            audio_ndarray_raw = frame.to_ndarray()
            
            # モノラルに変換（通常、[0] で最初のチャンネルを取得）
            if audio_ndarray_raw.ndim == 2:
                audio_mono = audio_ndarray_raw[0] # 最初のチャンネル (モノラル) を取得
            else:
                audio_mono = audio_ndarray_raw # もともとモノラルならそのまま

            # データ型をint16に変換（ParSelmouthはint16またはfloat64を好む）
            # PyAVのデフォルト出力がfloat32の場合があるため、適切にスケーリングしてint16に変換
            if audio_mono.dtype != np.int16:
                # floatをint16に変換する一般的な方法 (最大値を32767にスケーリング)
                # クリップを避けるため、floatの最大値1.0で割る
                audio_current_chunk = (audio_mono * 32767 / max(1.0, np.max(np.abs(audio_mono)))).astype(np.int16)
                # もし音声が非常に小さい場合、np.max(np.abs(audio_mono)) がゼロに近くなる可能性があるので、max(1.0, ...) でゼロ割を防ぐ
            else:
                audio_current_chunk = audio_mono # すでにint16ならそのまま
            
            # 現在のチャンクをバッファに追加
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_current_chunk))

            f1, f2 = float('nan'), float('nan') # 初期化

            # バッファが十分な長さに達したらフォルマントを分析
            # これにより、parselmouthに渡す音声区間が長くなり、安定した抽出が期待できます。
            if len(self.audio_buffer) >= self.buffer_size_samples:
                # 必要な長さだけを取り出して分析し、残りを次のバッファにする
                audio_for_analysis = self.audio_buffer[:self.buffer_size_samples]
                self.audio_buffer = self.audio_buffer[self.buffer_size_samples:] # 消費した分を削除

                # デバッグ情報
                print(f"DEBUG: Analyzing buffered chunk. Shape: {audio_for_analysis.shape}, Dtype: {audio_for_analysis.dtype}")
                print(f"DEBUG: Buffer remaining samples: {len(self.audio_buffer)}")

                f1, f2 = get_formants(audio_for_analysis, FS) # 長くなったチャンクを渡す
            
            # フォルマントが有効な数値かチェックし、履歴に追加・平均値を更新
            if is_valid(f1) and is_valid(f2):
                self.f1_history.append(f1)
                self.f2_history.append(f2)
                self.latest_f1 = f1
                self.latest_f2 = f2
                
                valid_f1s = [f for f in self.f1_history if is_valid(f)]
                valid_f2s = [f for f in self.f2_history if is_valid(f)]

                if valid_f1s:
                    self.output_dict["f1"] = np.mean(valid_f1s)
                else:
                    self.output_dict["f1"] = None
                
                if valid_f2s:
                    self.output_dict["f2"] = np.mean(valid_f2s)
                else:
                    self.output_dict["f2"] = None
                
                print(f"DEBUG: Formant data updated: F1={self.output_dict['f1']:.2f}, F2={self.output_dict['f2']:.2f}")
            else:
                print(f"DEBUG: Invalid formants received: F1={f1}, F2={f2}")

            return frame # 処理したフレームを返すことで、webrtc_streamerがエコーバックする
        except Exception as e:
            # recv メソッド全体でエラーをキャッチし、ログに出力
            print(f"ERROR in FormantProcessor.recv: {e}")
            return frame # エラーが発生してもフレームを返すことでストリームを維持

# --- Streamlit アプリのUI部分 ---

# 標準データ（男性・女性あいうえお）
reference_points = [
    ("man_a", 790, 1180),
    ("man_i", 250, 2300),
    ("man_u", 340, 1180),
    ("man_e", 460, 2060),
    ("man_o", 500, 800),
    ("woman_a", 950, 1450),
    ("woman_i", 290, 2930),
    ("woman_u", 400, 1430),
    ("woman_e", 590, 2430),
    ("woman_o", 610, 950),
]

st.title("リアルタイム母音フォルマントプロット")
st.write("マイク入力からリアルタイムでF1とF2フォルマントを抽出し、母音図にプロットします。")

# グラフの初期設定
fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim(200, 1200) # F1軸の範囲
ax.set_ylim(500, 3000) # F2軸の範囲
ax.set_xlabel("F1 (Hz)")
ax.set_ylabel("F2 (Hz)")
ax.set_title("F1-F2 母音図")
ax.invert_xaxis() # F1軸は通常反転
ax.invert_yaxis() # F2軸は通常反転

# 参照点をプロット
for label, f1, f2 in reference_points:
    color = "orange" if label.startswith("w") else "blue"
    ax.plot(f1, f2, 'o', color=color, markersize=6)
    ax.text(f1 + 10, f2 + 10, label, fontsize=9, color=color)

# 現在の音声の点をプロットするための初期化
point, = ax.plot([], [], 'ro', markersize=10, label="現在の音声")
ax.legend()

# グラフと情報表示用のプレースホルダー
chart_placeholder = st.empty()
info_placeholder = st.empty()

# webrtc_streamerを設定
webrtc_ctx = webrtc_streamer(
    key="speech_to_formant",
    audio_processor_factory=FormantProcessor,
    media_stream_constraints={
        "video": False, # ビデオは使用しない
        "audio": {
            "echoCancellation": False,   # エコーキャンセレーションを無効化
            "autoGainControl": False,    # 自動音量調整を無効化
            "noiseSuppression": False    # ノイズ抑制を無効化
        }
    },
    async_processing=False, # デバッグのため同期処理に設定 (重要)
)

# ストリームが開始されている場合
if webrtc_ctx.state.playing:
    info_placeholder.write("マイクがONになっています。話してみてください。")
    if webrtc_ctx.audio_processor:
        # Streamlitはスクリプトの再実行でUIを更新するため、
        # webrtc_ctx.audio_processor.output_dict の値が更新されるのを待つループが必要です。
        # async_processing=False の場合、recvはメインスレッドで呼ばれるため、
        # このポーリングループは CPU を消費します。しかし、デバッグのために一時的に使用します。
        # 本番では st.rerun() などをトリガーする方法を検討すると良いでしょう。
        while True:
            f1_val = webrtc_ctx.audio_processor.output_dict["f1"]
            f2_val = webrtc_ctx.audio_processor.output_dict["f2"]

            if f1_val is not None and f2_val is not None:
                point.set_xdata([f1_val])
                point.set_ydata([f2_val])
                chart_placeholder.pyplot(fig) # グラフを更新して表示
                info_placeholder.write(f"現在のF1: {f1_val:.2f} Hz, F2: {f2_val:.2f} Hz")
            else:
                info_placeholder.write("フォルマントデータを待機中...")
                chart_placeholder.pyplot(fig) # データがない場合でも初期グラフを表示
            
            time.sleep(0.05) # 少し待機してCPU負荷を軽減 (50ms)
    else:
        info_placeholder.write("音声処理を開始できませんでした。")
else:
    info_placeholder.write("マイク入力の準備ができていません。開始ボタンを押してください。")
    chart_placeholder.pyplot(fig) # 開始ボタンが押される前も初期グラフを表示

st.write("---")
st.info("このアプリは実験的なものです。フォルマント抽出の精度は、マイクの品質、発声方法、環境ノイズに影響されます。")
st.markdown("参考: [streamlit-webrtc](https://github.com/whitenoise/streamlit-webrtc)")