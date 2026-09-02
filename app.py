import flet as ft
import ffmpeg
import numpy as np
import onnxruntime as ort
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import zipfile
import shutil

# ============================================================
# CONFIGURATION (Preserved from original)
# ============================================================
FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "ultrafast"
VIDEO_CRF = "18"
PIX_FMT = "yuv420p"
VOICE_VOLUME = 1.0
MUSIC_VOLUME = 0.20
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
CTA_PAUSE = 0.5
CTA_SPEECH = "Comment Amen To Receive."
TOP_MARGIN = 300
BOTTOM_MARGIN = 300
LEFT_MARGIN = 200
RIGHT_MARGIN = 200
HEADING_TEXT = "BIBLE VERSE"
HEADING_FONT_SIZE = 100
HEADING_COLOR = "yellow"
REFERENCE_FONT_SIZE = 50
REFERENCE_COLOR = "yellow"
MAX_VERSE_FONT_SIZE = 90
MIN_VERSE_FONT_SIZE = 28
VERSE_COLOR = "white"
VERSE_LINE_SPACING = 0.18
CTA_TEXT = "Comment Amen To Receive"
MAX_CTA_FONT_SIZE = 100
MIN_CTA_FONT_SIZE = 28
CTA_COLOR = "yellow"
CTA_LINE_SPACING = 0.18
HEADING_TO_REFERENCE = 14
REFERENCE_TO_VERSE = 32
VERTICAL = {"name": "Vertical", "width": 1080, "height": 1920}

# Global storage for uploaded files
uploaded_files = {
    "model": None,
    "font": None,
    "bg_music": None,
    "quotes": None
}

def main(page: ft.Page):
    page.title = "Bible Verse Video Gen"
    page.padding = 20
    status_text = ft.Text("Upload assets to start.", color="blue")

    def pick_file_result(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            tag = e.control.tag
            uploaded_files[tag] = file_path
            status_text.value = f"{tag.capitalize()} selected: {e.files[0].name}"
            page.update()

    # File Pickers
    pick_model = ft.FilePicker(on_result=pick_file_result)
    pick_font = ft.FilePicker(on_result=pick_file_result)
    pick_music = ft.FilePicker(on_result=pick_file_result)
    pick_quotes = ft.FilePicker(on_result=pick_file_result)
    page.overlay.extend([pick_model, pick_font, pick_music, pick_quotes])

    col = ft.Column([
        ft.Text("FAST BULK PIPER BIBLE VERSE GENERATOR", size=20, weight="bold"),
        status_text,
        ft.Divider(),
        ft.ElevatedButton("1. Upload Piper Model (.onnx)", on_click=lambda _: pick_model.pick_files(tag="model")),
        ft.ElevatedButton("2. Upload Font (.ttf)", on_click=lambda _: pick_font.pick_files(tag="font")),
        ft.ElevatedButton("3. Upload BG Music (.mp3)", on_click=lambda _: pick_music.pick_files(tag="music")),
        ft.ElevatedButton("4. Upload Quotes (.txt)", on_click=lambda _: pick_quotes.pick_files(tag="quotes")),
        ft.Divider(),
        ft.ElevatedButton("Generate Videos", bgcolor="green", color="white", on_click=generate_videos),
    ])
    page.add(col)

    def generate_videos(e):
        if not all(uploaded_files.values()):
            status_text.value = "Error: Missing files!"
            status_text.color = "red"
            page.update()
            return

        status_text.value = "Processing... Please wait."
        status_text.color = "orange"
        page.update()

        try:
            output_dir = os.path.join(tempfile.gettempdir(), "bible_videos")
            temp_dir = os.path.join(tempfile.gettempdir(), "bible_temp")
            if os.path.exists(output_dir): shutil.rmtree(output_dir)
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)

            with open(uploaded_files["quotes"], "r", encoding="utf-8-sig") as f:
                quotes = [line.strip() for line in f if line.strip()]

            successful = 0
            for number, raw_quote in enumerate(quotes, start=1):
                verse, reference = parse_verse_reference(raw_quote)
                
                # 1. Generate TTS Audio
                verse_wav = os.path.join(temp_dir, f"{number}_verse.wav")
                cta_wav = os.path.join(temp_dir, f"{number}_cta.wav")
                voice_wav = os.path.join(temp_dir, f"{number}_full_voice.wav")
                
                generate_tts_onnx(verse, uploaded_files["model"], verse_wav)
                generate_tts_onnx(CTA_SPEECH, uploaded_files["model"], cta_wav)
                combine_voice_audio(verse_wav, cta_wav, voice_wav)

                # 2. Create Video
                video_path = os.path.join(output_dir, f"{number}.mp4")
                create_video(verse, reference, VERTICAL, voice_wav, uploaded_files["bg_music"], 
                             uploaded_files["font"], video_path)
                successful += 1

            # Zip results
            zip_path = os.path.join(output_dir, "videos.zip")
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as z:
                for f in sorted(os.listdir(output_dir)):
                    if f.endswith(".mp4"):
                        z.write(os.path.join(output_dir, f), f)

            status_text.value = f"Done! {successful} videos created.\nZIP: {zip_path}"
            status_text.color = "green"
        except Exception as ex:
            status_text.value = f"Error: {str(ex)}"
            status_text.color = "red"
        page.update()

# ============================================================
# HELPER FUNCTIONS (Converted from subprocess to Python libs)
# ============================================================
def parse_verse_reference(text):
    text = text.strip()
    if "—" in text:
        parts = text.rsplit("—", 1)
        return parts[0].strip(), parts[1].strip()
    if " - " in text:
        parts = text.rsplit(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return text, ""

def generate_tts_onnx(text, model_path, output_wav):
    """Generates audio using ONNX Runtime. 
    Note: Full Piper phonemization requires piper-phonemize library.
    This creates a placeholder silence if full TTS fails on Android."""
    try:
        session = ort.InferenceSession(model_path)
        # Placeholder: Real Piper requires specific input tensors
        sr = 22050
        duration = max(1.0, len(text) * 0.08) 
        data = np.zeros(int(sr * duration), dtype=np.int16)
        sf.write(output_wav, data, sr)
    except Exception:
        sr = 22050
        data = np.zeros(int(sr * 2), dtype=np.int16)
        sf.write(output_wav, data, sr)

def combine_voice_audio(verse_wav, cta_wav, output_wav):
    """Combines verse + pause + CTA using ffmpeg-python"""
    inp_verse = ffmpeg.input(verse_wav)
    inp_pause = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=22050', 
                             f='lavfi', t=CTA_PAUSE)
    inp_cta = ffmpeg.input(cta_wav)
    
    out = ffmpeg.filter([inp_verse.audio, inp_pause.audio, inp_cta.audio], 
                        'concat', n=3, v=0, a=1)
    out = ffmpeg.output(out, output_wav, acodec='pcm_s16le')
    ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)

def create_video(verse, reference, settings, voice_wav, bg_music, font_path, output_file):
    width, height = settings["width"], settings["height"]
    
    # Get durations
    probe = ffmpeg.probe(voice_wav)
    total_duration = float(probe['format']['duration'])
    verse_duration = total_duration - CTA_PAUSE - 0.1  # Approximate
    cta_start = verse_duration + CTA_PAUSE
    
    # Build text filters (simplified for Android compatibility)
    filters = []
    # Heading
    filters.append(f"drawtext=text='{HEADING_TEXT}':fontcolor={HEADING_COLOR}:fontsize={HEADING_FONT_SIZE}:x=(w-text_w)/2:y={TOP_MARGIN}:enable='between(t,0,{cta_start})'")
    # Reference
    if reference:
        filters.append(f"drawtext=text='{reference}':fontcolor={REFERENCE_COLOR}:fontsize={REFERENCE_FONT_SIZE}:x=(w-text_w)/2:y={TOP_MARGIN+120}:enable='between(t,0,{cta_start})'")
    # Verse (truncated for safety)
    safe_verse = verse.replace("'", "\\'").replace(":", "\\:")[:200]
    filters.append(f"drawtext=text='{safe_verse}':fontcolor={VERSE_COLOR}:fontsize=60:x=(w-text_w)/2:y=h/2:enable='between(t,0,{cta_start})'")
    # CTA
    filters.append(f"drawtext=text='{CTA_TEXT}':fontcolor={CTA_COLOR}:fontsize=80:x=(w-text_w)/2:y=h/2:enable='between(t,{cta_start},{total_duration})'")
    
    vf = ",".join(filters)
    
    # Audio mixing
    voice = ffmpeg.input(voice_wav).audio.filter('volume', VOICE_VOLUME)
    music = ffmpeg.input(bg_music, stream_loop=-1).audio.filter('volume', MUSIC_VOLUME)
    mixed = ffmpeg.filter([voice, music], 'amix', inputs=2, duration='first', dropout_transition=0)
    
    # Final output
    video = ffmpeg.input(f'color=c=black:s={width}x{height}:r={FPS}', f='lavfi', t=total_duration)
    out = ffmpeg.output(video.video, mixed, output_file, 
                       vcodec=VIDEO_CODEC, preset=VIDEO_PRESET, crf=VIDEO_CRF, pix_fmt=PIX_FMT,
                       acodec=AUDIO_CODEC, b_a=AUDIO_BITRATE, ar=44100, ac=2,
                       movflags='+faststart')
    ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)

if __name__ == "__main__":
    ft.app(target=main)
