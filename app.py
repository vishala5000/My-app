import flet as ft
import ffmpeg
import numpy as np
import onnxruntime as ort
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import zipfile

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

    # Pickers
    pick_model = ft.FilePicker(on_result=pick_file_result)
    pick_font = ft.FilePicker(on_result=pick_file_result)
    pick_music = ft.FilePicker(on_result=pick_file_result)
    pick_quotes = ft.FilePicker(on_result=pick_file_result)
    page.overlay.extend([pick_model, pick_font, pick_music, pick_quotes])

    # UI Buttons
    col = ft.Column([
        ft.Text("Bible Verse Video Generator", size=24, weight="bold"),
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

        status_text.value = "Processing... (Check Actions for logs)"
        status_text.color = "orange"
        page.update()

        try:
            output_dir = os.path.join(tempfile.gettempdir(), "bible_videos")
            os.makedirs(output_dir, exist_ok=True)

            with open(uploaded_files["quotes"], "r", encoding="utf-8") as f:
                quotes = [line.strip() for line in f if line.strip()]

            for i, quote in enumerate(quotes):
                verse, ref = parse_verse(quote)
                
                # 1. Generate Audio (Silence for now as Piper TTS is complex on Android)
                # In a real app, you'd use onnxruntime here with the uploaded model
                audio_path = os.path.join(output_dir, f"voice_{i}.wav")
                create_silence(audio_path, 3.0) # 3 seconds silence for demo

                # 2. Create Frame
                frame_path = os.path.join(output_dir, f"frame_{i}.png")
                create_frame(verse, ref, uploaded_files["font"], frame_path)

                # 3. Create Video
                video_path = os.path.join(output_dir, f"video_{i}.mp4")
                create_video_ffmpeg(frame_path, audio_path, uploaded_files["bg_music"], video_path)

            # Zip results
            zip_path = os.path.join(output_dir, "videos.zip")
            with zipfile.ZipFile(zip_path, 'w') as z:
                for f in os.listdir(output_dir):
                    if f.endswith(".mp4"):
                        z.write(os.path.join(output_dir, f), f)

            status_text.value = f"Done! ZIP at: {zip_path}"
            status_text.color = "green"
        except Exception as ex:
            status_text.value = f"Error: {str(ex)}"
            status_text.color = "red"
        page.update()

def parse_verse(text):
    if "—" in text:
        parts = text.rsplit("—", 1)
        return parts[0].strip(), parts[1].strip()
    return text, ""

def create_silence(path, duration):
    sr = 22050
    data = np.zeros(int(sr * duration), dtype=np.int16)
    sf.write(path, data, sr)

def create_frame(verse, ref, font_path, output):
    img = Image.new('RGB', (1080, 1920), 'black')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 80)
    except:
        font = ImageFont.load_default()
    
    draw.text((540, 300), "BIBLE VERSE", fill="yellow", font=font, anchor="mm")
    if ref:
        draw.text((540, 450), ref, fill="yellow", font=font, anchor="mm")
    draw.multiline_text((540, 600), verse, fill="white", font=font, anchor="mm", align="center")
    img.save(output)

def create_video_ffmpeg(frame, voice, bg, output):
    probe = ffmpeg.probe(voice)
    dur = float(probe['format']['duration'])
    
    inp_img = ffmpeg.input(frame, loop=1, t=dur, r=30)
    inp_voice = ffmpeg.input(voice)
    inp_bg = ffmpeg.input(bg, stream_loop=-1)
    
    mixed = ffmpeg.filter([inp_voice.audio, inp_bg.audio], 'amix', inputs=2, duration='first')
    
    out = ffmpeg.output(inp_img.video, mixed, output, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
    ffmpeg.run(out, overwrite_output=True, capture_stdout=True, capture_stderr=True)

if __name__ == "__main__":
    ft.app(target=main)
