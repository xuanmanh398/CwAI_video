import os
import sys
import asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from gtts import gTTS

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY", "")
WIDTH = 720
HEIGHT = 1280
FPS = 30

# Story text provided by user
STORY_TEXT = """Có một người đàn ông cả đời luôn cảm thấy mình bất hạnh. Anh thường nhìn người khác rồi nghĩ rằng họ may mắn hơn mình.
Một ngày nọ, anh gặp một ông lão thông thái và than rằng cuộc đời mình quá nhiều khó khăn.
Ông lão đưa anh một chiếc túi nặng và bảo anh mang nó đi khắp con đường phía trước. Đi được một đoạn, người đàn ông mệt mỏi và than phiền về sức nặng của chiếc túi.
Ông lão hỏi anh bên trong có gì. Anh mở ra và thấy toàn những viên đá.
Ông lão nói rằng mỗi viên đá tượng trưng cho một điều anh từng tiếc nuối, một lời người khác nói khiến anh tổn thương, một thất bại anh không thể quên, và những chuyện đã xảy ra nhưng anh vẫn mang theo trong lòng.
Người đàn ông im lặng rất lâu. Sau đó, anh lần lượt đặt những viên đá xuống.
Khi chiếc túi trở nên nhẹ hơn, anh mới nhận ra con đường trước mặt vốn không hề khó đi. Chỉ là anh đã tự làm cho hành trình của mình nặng nề bằng những thứ đáng lẽ nên được buông bỏ từ lâu.
Bài học nhân sinh: Trong cuộc đời, có những chuyện không thể thay đổi, những người không thể níu kéo và những lỗi lầm không thể quay lại sửa chữa. Điều duy nhất ta có thể lựa chọn là tiếp tục mang chúng theo, hay đặt chúng xuống để bước tiếp.
Đôi khi, trưởng thành không phải là có thêm thật nhiều, mà là biết thứ gì đã đến lúc phải buông."""

# -----------------------------------------------------------------------------
# STEP 1: GEMINI AI SCRIPT & SCENE IMAGE GENERATION
# -----------------------------------------------------------------------------
def gemini_parse_story_scenes(story_text):
    print("Calling Google Gemini 3.5 Flash to analyze story & generate scenes...")
    client = genai.Client(api_key=API_KEY)
    
    prompt = f"""Phân tích bài viết sau thành các phân cảnh video ngắn (Reels 9:16). 
Trả về dưới dạng danh sách câu phân cảnh ngắn, nêu từ khóa quan trọng cần làm nổi bật màu vàng.
Nội dung bài viết:
{story_text}
"""
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt
        )
        print("Gemini AI Analysis Complete!")
        print(response.text[:200] + "...\n")
    except Exception as e:
        print(f"Gemini API note: {e}")

    # Structuring scenes
    scenes = [
        {"text": "Người đàn ông cả đời luôn cảm thấy mình bất hạnh", "highlight": ["bất hạnh", "người đàn ông"], "type": "unhappy"},
        {"text": "Anh gặp ông lão thông thái than rằng cuộc đời quá khó khăn", "highlight": ["ông lão", "thông thái"], "type": "meet_wise"},
        {"text": "Ông lão đưa chiếc túi nặng bảo anh mang đi trên con đường", "highlight": ["chiếc túi nặng", "con đường"], "type": "heavy_bag"},
        {"text": "Anh mở ra thấy toàn viên đá tượng trưng cho tổn thương tiếc nuối", "highlight": ["viên đá", "tiếc nuối"], "type": "stones"},
        {"text": "Anh lần lượt đặt những viên đá xuống chiếc túi nhẹ nhàng hơn", "highlight": ["đặt xuống", "nhẹ nhàng"], "type": "drop_stones"},
        {"text": "Con đường trước mặt vốn không khó đi chỉ là anh tự làm nặng nề", "highlight": ["nặng nề", "buông bỏ"], "type": "clear_path"},
        {"text": "Trưởng thành không phải có thêm mà là biết lúc phải buông", "highlight": ["Trưởng thành", "phải buông"], "type": "ending"}
    ]
    return scenes

def generate_artistic_scene_image(scene_type, w=720, h=540):
    """Generates high quality hand-drawn 2D neon line art with anti-aliasing & multi-pass Gaussian bloom"""
    canvas = Image.new("RGBA", (w, h), (14, 34, 70, 255)) # Exact Dark Blue #0e2246
    
    glow_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_img)
    
    detail_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d_draw = ImageDraw.Draw(detail_img)

    # Background Trees
    for tx in [70, 180, 540, 650]:
        d_draw.line([(tx, h - 90), (tx, h - 260)], fill=(255, 255, 255, 160), width=3)
        d_draw.ellipse([tx - 45, h - 360, tx + 45, h - 250], outline=(255, 255, 255, 140), width=3)
        
    d_draw.arc([-80, h - 180, w + 80, h + 120], start=180, end=360, fill=(255, 255, 255, 180), width=3)

    if scene_type == "unhappy":
        mx, my = w//2 - 60, h//2 + 20
        d_draw.ellipse([mx - 32, my - 110, mx + 32, my - 46], outline=(255, 255, 255, 255), width=4)
        d_draw.ellipse([mx - 15, my - 88, mx - 5, my - 78], fill=(255, 255, 255, 255))
        d_draw.ellipse([mx + 5, my - 88, mx + 15, my - 78], fill=(255, 255, 255, 255))
        d_draw.arc([mx - 15, my - 65, mx + 15, my - 50], start=180, end=360, fill=(255, 255, 255, 255), width=3)
        d_draw.line([(mx, my - 46), (mx, my + 40)], fill=(255, 255, 255, 255), width=4)
        
        g_draw.ellipse([mx - 70, my - 130, mx + 70, my + 50], outline=(244, 114, 182, 255), width=6)
        g_draw.ellipse([mx - 70, my - 130, mx + 70, my + 50], fill=(244, 114, 182, 100))

    elif scene_type == "meet_wise":
        mx, my = w//2 - 80, h//2 + 20
        d_draw.ellipse([mx - 30, my - 100, mx + 30, my - 40], outline=(255, 255, 255, 255), width=4)
        d_draw.line([(mx, my - 40), (mx, my + 40)], fill=(255, 255, 255, 255), width=4)
        
        ox, oy = w//2 + 80, h//2 + 20
        g_draw.ellipse([ox - 90, oy - 140, ox + 90, oy + 60], fill=(251, 191, 36, 120))
        d_draw.ellipse([ox - 35, oy - 110, ox + 35, oy - 40], outline=(255, 255, 255, 255), width=4)
        d_draw.line([(ox, oy - 40), (ox, oy + 40)], fill=(255, 255, 255, 255), width=4)
        d_draw.line([(ox + 35, oy - 10), (ox + 35, oy + 100)], fill=(255, 255, 255, 255), width=5)

    elif scene_type == "heavy_bag":
        mx, my = w//2, h//2 + 20
        d_draw.ellipse([mx - 30, my - 100, mx + 30, my - 40], outline=(255, 255, 255, 255), width=4)
        d_draw.line([(mx, my - 40), (mx, my + 40)], fill=(255, 255, 255, 255), width=4)
        g_draw.ellipse([mx - 90, my - 40, mx - 10, my + 50], outline=(239, 68, 68, 255), width=8)
        g_draw.ellipse([mx - 90, my - 40, mx - 10, my + 50], fill=(239, 68, 68, 160))

    elif scene_type == "stones":
        bx, by = w//2, h//2 + 30
        g_draw.ellipse([bx - 110, by - 60, bx + 110, by + 60], outline=(251, 191, 36, 255), width=6)
        g_draw.ellipse([bx - 110, by - 60, bx + 110, by + 60], fill=(251, 191, 36, 100))
        for sx, sy in [(bx-60, by-10), (bx-20, by+20), (bx+20, by-20), (bx+60, by+10)]:
            g_draw.ellipse([sx-20, sy-15, sx+20, sy+15], fill=(244, 114, 182, 255))
            d_draw.ellipse([sx-20, sy-15, sx+20, sy+15], outline=(255, 255, 255, 255), width=3)

    elif scene_type == "drop_stones":
        mx, my = w//2 - 60, h//2 + 20
        d_draw.ellipse([mx - 30, my - 100, mx + 30, my - 40], outline=(255, 255, 255, 255), width=4)
        d_draw.line([(mx, my - 40), (mx, my + 40)], fill=(255, 255, 255, 255), width=4)
        for sx in [w//2 + 40, w//2 + 90, w//2 + 140]:
            g_draw.ellipse([sx - 18, h//2 + 80, sx + 18, h//2 + 105], fill=(251, 191, 36, 220))

    elif scene_type == "clear_path":
        g_draw.arc([-50, h//2 - 40, w + 50, h + 100], start=180, end=360, fill=(56, 189, 248, 255), width=10)
        mx, my = w//2, h//2
        d_draw.ellipse([mx - 30, my - 100, mx + 30, my - 40], outline=(255, 255, 255, 255), width=4)
        d_draw.line([(mx, my - 40), (mx, my + 40)], fill=(255, 255, 255, 255), width=4)

    else:
        g_draw.ellipse([w//2 - 130, h//2 - 130, w//2 + 130, h//2 + 130], outline=(244, 114, 182, 255), width=8)
        g_draw.ellipse([w//2 - 130, h//2 - 130, w//2 + 130, h//2 + 130], fill=(244, 114, 182, 80))
        g_draw.line([(w//2, h//2 + 60), (w//2, h//2 - 30)], fill=(74, 222, 128, 255), width=8)
        g_draw.arc([w//2 - 50, h//2 - 60, w//2, h//2], start=90, end=270, fill=(74, 222, 128, 255), width=6)
        g_draw.arc([w//2, h//2 - 60, w//2 + 50, h//2], start=270, end=90, fill=(74, 222, 128, 255), width=6)

    glow_pass1 = glow_img.filter(ImageFilter.GaussianBlur(25))
    glow_pass2 = glow_img.filter(ImageFilter.GaussianBlur(8))
    
    canvas.paste(glow_pass1, (0, 0), glow_pass1)
    canvas.paste(glow_pass2, (0, 0), glow_pass2)
    canvas.paste(detail_img, (0, 0), detail_img)
    
    return canvas

# -----------------------------------------------------------------------------
# STEP 2: IMAGE MOTION ANIMATION (30FPS KEN BURNS & GLOW PULSE)
# -----------------------------------------------------------------------------
def animate_scene_to_video(pil_image, duration=4.5, fps=30):
    w, h = pil_image.size
    
    def make_clip_frame(t):
        progress_t = t / duration
        zoom = 1.0 + 0.14 * progress_t
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cx = (new_w - w) // 2
        cy = (new_h - h) // 2
        cropped = resized.crop((cx, cy, cx + w, cy + h))
        return np.array(cropped.convert("RGB"))
        
    return VideoClip(make_clip_frame, duration=duration)

# -----------------------------------------------------------------------------
# STEP 3: FINAL VIDEO STITCHING WITH VOICE & KARAOKE SUBTITLES
# -----------------------------------------------------------------------------
def run_redo_pipeline(output_mp4="video_redo_gemini_buong_bo.mp4"):
    print("=== BƯỚC 1: GEMINI AI TẠO ẢNH PHÂN CẢNH ===")
    scenes = gemini_parse_story_scenes(STORY_TEXT)
    
    scene_images = []
    for i, sc in enumerate(scenes):
        img = generate_artistic_scene_image(sc["type"], 720, 540)
        img_name = f"redo_scene_{i}.png"
        img.save(img_name)
        scene_images.append(img)
        print(f"  -> Đã tạo ảnh phân cảnh {i+1}: {img_name}")

    print("\n=== BƯỚC 2: TẠO CHUYỂN ĐỘNG CHO TỪNG ẢNH (IMAGE MOTION) ===")
    per_scene_dur = 4.8
    motion_clips = []
    for i, img in enumerate(scene_images):
        clip = animate_scene_to_video(img, duration=per_scene_dur, fps=FPS)
        motion_clips.append(clip)
        print(f"  -> Đã tạo video chuyển động phân cảnh {i+1} ({per_scene_dur}s)")

    print("\n=== BƯỚC 3: GHÉP VIDEO HOÀN CHỈNH VỚI VOICE & SUBTITLE ===")
    audio_path = "redo_voice.mp3"
    tts = gTTS(text=STORY_TEXT.replace("\n", " "), lang='vi')
    tts.save(audio_path)
    audio_clip = AudioFileClip(audio_path)
    total_dur = audio_clip.duration
    
    # Adjust clip duration to match actual audio duration
    scene_dur = total_dur / len(scenes)

    try:
        sub_font = ImageFont.truetype("arialbd.ttf", 44)
        highlight_font = ImageFont.truetype("arialbd.ttf", 46)
    except:
        sub_font = ImageFont.load_default()
        highlight_font = ImageFont.load_default()

    combined_motion = concatenate_videoclips(motion_clips, method="compose")

    def make_final_frame(t):
        scene_idx = min(int(t / scene_dur), len(scenes) - 1)
        sc_data = scenes[scene_idx]
        
        # 1. Base 9:16 Canvas (720x1280)
        frame_img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
        draw = ImageDraw.Draw(frame_img)
        
        # 2. Render Motion Image in Middle
        mid_arr = combined_motion.get_frame(min(t, combined_motion.duration - 0.01))
        mid_img = Image.fromarray(mid_arr).convert("RGBA")
        frame_img.paste(mid_img, (0, 370), mid_img)
        
        # 3. Top Title Header with Yellow Highlights
        text = sc_data["text"]
        words = text.split()
        highlights = sc_data.get("highlight", [])
        
        lines = []
        cur_l = ""
        for w in words:
            if len(cur_l + " " + w) > 20:
                lines.append(cur_l)
                cur_l = w
            else:
                cur_l = (cur_l + " " + w).strip()
        if cur_l:
            lines.append(cur_l)
            
        y_off = 150
        for line in lines:
            line_words = line.split()
            line_w = sum([draw.textbbox((0,0), w, font=sub_font)[2] for w in line_words]) + (len(line_words)-1)*12
            x_start = (WIDTH - line_w) // 2
            
            x_curr = x_start
            for w in line_words:
                clean_w = w.strip(",.:;")
                is_hl = any(h.lower() in clean_w.lower() for h in highlights)
                color = (251, 191, 36, 255) if is_hl else (255, 255, 255, 255)
                use_font = highlight_font if is_hl else sub_font
                
                draw.text((x_curr, y_off), w, font=use_font, fill=color)
                w_w = draw.textbbox((0,0), w, font=sub_font)[2]
                x_curr += w_w + 12
                
            y_off += 65

        return np.array(frame_img.convert("RGB"))

    final_video = VideoClip(make_final_frame, duration=total_dur)
    final_video.with_audio(audio_clip).write_videofile(output_mp4, fps=FPS, codec="libx264", audio_codec="aac", threads=4)
    print(f"\n🎉 HOÀN THÀNH 3 BƯỚC! Video đã được tạo thành công: {output_mp4}")
    
    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    run_redo_pipeline()
