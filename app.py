import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from moviepy.editor import ImageSequenceClip
import tempfile
import os
import io
from pathlib import Path
import time
from scipy import ndimage

class StreamlitImageAnimator:
    def __init__(self):
        self.effects = {
            'bounce': self.create_bounce_effect,
            'rotate': self.create_rotate_effect,
            'zoom': self.create_zoom_effect,
            'wave': self.create_wave_effect,
            'fade': self.create_fade_effect,
            'shake': self.create_shake_effect
        }

    def create_bounce_effect(self, img_array, frames=30, intensity=1.0):
        """Create a bouncing animation effect"""
        height, width = img_array.shape[:2]
        result = []
        
        for i in range(frames):
            # Calculate vertical displacement
            displacement = int(20 * intensity * abs(np.sin(2 * np.pi * i / frames)))
            frame = np.zeros_like(img_array)
            
            # Apply displacement
            if displacement > 0:
                frame[displacement:, :] = img_array[:-displacement, :]
            else:
                frame = img_array.copy()
                
            result.append(frame)
        
        return result

    def create_rotate_effect(self, img_array, frames=30, intensity=1.0):
        """Create a rotating animation effect"""
        result = []
        
        for i in range(frames):
            # Calculate rotation angle
            angle = intensity * 360 * i / frames
            frame = ndimage.rotate(img_array, angle, reshape=False)
            result.append(frame)
            
        return result

    def create_zoom_effect(self, img_array, frames=30, intensity=1.0):
        """Create a zoom pulse animation effect"""
        height, width = img_array.shape[:2]
        result = []
        
        for i in range(frames):
            # Calculate zoom factor
            zoom = 1 + 0.2 * intensity * np.sin(2 * np.pi * i / frames)
            frame = ndimage.zoom(img_array, [zoom, zoom, 1] if len(img_array.shape) > 2 else [zoom, zoom])
            
            # Crop to original size
            y, x = frame.shape[:2]
            start_y = (y - height) // 2
            start_x = (x - width) // 2
            frame = frame[start_y:start_y+height, start_x:start_x+width]
            
            result.append(frame)
            
        return result

    def create_wave_effect(self, img_array, frames=30, intensity=1.0):
        """Create a wave animation effect"""
        height, width = img_array.shape[:2]
        result = []
        
        for i in range(frames):
            frame = np.zeros_like(img_array)
            for x in range(width):
                offset = int(10 * intensity * np.sin(2 * np.pi * (x/width + i/frames)))
                for y in range(height):
                    new_y = (y + offset) % height
                    frame[new_y, x] = img_array[y, x]
            result.append(frame)
            
        return result

    def create_fade_effect(self, img_array, frames=30, intensity=1.0):
        """Create a fade in/out effect"""
        result = []
        
        for i in range(frames):
            # Calculate opacity
            opacity = abs(np.sin(2 * np.pi * i / frames))
            frame = (img_array * opacity).astype(np.uint8)
            result.append(frame)
            
        return result

    def create_shake_effect(self, img_array, frames=30, intensity=1.0):
        """Create a shaking animation effect"""
        height, width = img_array.shape[:2]
        result = []
        
        for i in range(frames):
            # Random displacement
            dx = int(10 * intensity * np.random.randn())
            dy = int(10 * intensity * np.random.randn())
            
            frame = np.zeros_like(img_array)
            
            # Ensure displacement doesn't exceed image boundaries
            if dx > 0:
                frame[:, dx:] = img_array[:, :-dx]
            else:
                frame[:, :dx] = img_array[:, -dx:]
                
            if dy > 0:
                frame[dy:, :] = frame[:-dy, :]
            else:
                frame[:dy, :] = frame[-dy:, :]
                
            result.append(frame)
            
        return result

def process_image(uploaded_file, max_size=800):
    """Process uploaded image with size limitation"""
    try:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        ratio = min(max_size / image.size[0], max_size / image.size[1])
        if ratio < 1:
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return np.array(image), None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def main():
    st.set_page_config(
        page_title="Animated Image Creator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎨 Animated Image Creator")
    st.markdown("Create stunning animations from your images!")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        animation_type = st.selectbox(
            "Animation Effect",
            ["bounce", "rotate", "zoom", "wave", "fade", "shake"]
        )
        
        with st.expander("Advanced Settings"):
            fps = st.slider("Frames Per Second", 10, 60, 30)
            duration = st.slider("Duration (seconds)", 1, 5, 2)
            intensity = st.slider("Effect Intensity", 0.1, 2.0, 1.0)

    # Main content
    col1, col2 = st.columns([2, 3])

    with col1:
        if uploaded_file:
            st.subheader("Original Image")
            image, error = process_image(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.image(image, use_column_width=True)

    with col2:
        if uploaded_file and image is not None:
            st.subheader("Animation Preview")
            
            if st.button("Generate Animation", type="primary"):
                with st.spinner("Creating animation..."):
                    try:
                        # Create animation
                        animator = StreamlitImageAnimator()
                        frames = animator.effects[animation_type](
                            image,
                            frames=duration * fps,
                            intensity=intensity
                        )
                        
                        # Create video
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                            clip = ImageSequenceClip(frames, fps=fps)
                            clip.write_videofile(
                                tmp_file.name,
                                fps=fps,
                                codec='libx264',
                                audio=False,
                                verbose=False,
                                logger=None
                            )
                            
                            # Read the video file
                            with open(tmp_file.name, 'rb') as f:
                                video_bytes = f.read()
                            
                            # Clean up
                            os.unlink(tmp_file.name)
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'effect': animation_type,
                            'video': video_bytes
                        })
                        
                        # Display video and download button
                        st.video(video_bytes)
                        st.download_button(
                            "Download Animation",
                            video_bytes,
                            f"animation_{animation_type}.mp4",
                            "video/mp4"
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # History section
    if st.session_state.history:
        st.header("Animation History")
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Animation {len(st.session_state.history)-idx}: {item['effect']} ({item['timestamp']})"):
                st.video(item['video'])
                st.download_button(
                    f"Download Animation {len(st.session_state.history)-idx}",
                    item['video'],
                    f"animation_{item['effect']}_{idx}.mp4",
                    "video/mp4"
                )

    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        ### Instructions
        1. Upload an image using the sidebar
        2. Choose an animation effect
        3. Adjust advanced settings if desired
        4. Click 'Generate Animation' to create your animation
        5. Download the result or browse your animation history
        
        ### Available Effects
        - **Bounce**: Creates a bouncing motion
        - **Rotate**: Spins the image
        - **Zoom**: Pulsing zoom effect
        - **Wave**: Rippling wave animation
        - **Fade**: Smooth fade in/out
        - **Shake**: Random shaking motion
        
        ### Tips
        - Larger images will be automatically resized for better performance
        - Higher FPS creates smoother animations but takes longer to process
        - Adjust the intensity slider to control the strength of the effect
        """)

if __name__ == "__main__":
    main()
