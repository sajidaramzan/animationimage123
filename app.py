import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from moviepy.editor import ImageSequenceClip
import tempfile
import os
import io
from pathlib import Path
import time

class AdvancedImageAnimator:
    def __init__(self):
        self.effects = {
            'walking': self.create_walking_effect,
            'working': self.create_working_effect,
            'typing': self.create_typing_effect,
            'nodding': self.create_nodding_effect,
            'floating': self.create_floating_effect,
            'spinning': self.create_spinning_effect,
            'pulse': self.create_pulse_effect
        }
        
    def apply_image_filters(self, image, filters):
        """Apply various image filters"""
        img = Image.fromarray(image)
        
        if filters.get('brightness'):
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(filters['brightness'])
            
        if filters.get('contrast'):
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(filters['contrast'])
            
        if filters.get('blur'):
            img = img.filter(ImageFilter.GaussianBlur(filters['blur']))
            
        if filters.get('sharpen'):
            img = img.filter(ImageFilter.SHARPEN)
            
        return np.array(img)

    def create_walking_effect(self, image, frames=60, intensity=1.0):
        """Enhanced walking animation effect"""
        height, width = image.shape[:2]
        result = []
        
        for i in range(frames):
            frame = image.copy()
            # Enhanced side-to-side movement
            shift = int(10 * intensity * np.sin(2 * np.pi * i / 30))
            vertical_shift = int(5 * intensity * np.sin(2 * np.pi * i / 15))
            
            # Create transformation matrix
            M = np.float32([
                [1, 0, shift],
                [0, 1, vertical_shift]
            ])
            
            # Apply transformation
            frame = self._apply_affine_transform(frame, M)
            result.append(frame)
        
        return result

    def create_floating_effect(self, image, frames=60, intensity=1.0):
        """Create floating/hovering animation effect"""
        height, width = image.shape[:2]
        result = []
        
        for i in range(frames):
            frame = image.copy()
            # Combined vertical and slight rotation movement
            vertical_shift = int(10 * intensity * np.sin(2 * np.pi * i / 30))
            angle = 2 * intensity * np.sin(2 * np.pi * i / 60)
            
            # Create transformation matrix
            M = np.float32([
                [np.cos(angle), -np.sin(angle), width * 0.1 * np.sin(angle)],
                [np.sin(angle), np.cos(angle), vertical_shift]
            ])
            
            # Apply transformation
            frame = self._apply_affine_transform(frame, M)
            result.append(frame)
        
        return result

    def create_spinning_effect(self, image, frames=60, intensity=1.0):
        """Create spinning animation effect"""
        height, width = image.shape[:2]
        result = []
        
        for i in range(frames):
            # Calculate rotation angle
            angle = (360 * i / frames) * intensity
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, M, (width, height))
            result.append(rotated)
            
        return result

    def create_pulse_effect(self, image, frames=60, intensity=1.0):
        """Create pulsing/scaling animation effect"""
        height, width = image.shape[:2]
        result = []
        
        for i in range(frames):
            # Calculate scale factor
            scale = 1 + 0.1 * intensity * np.sin(2 * np.pi * i / 30)
            
            # Create scaling matrix
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
            
            # Apply scaling
            scaled = cv2.warpAffine(image, M, (width, height))
            result.append(scaled)
            
        return result

    def _apply_affine_transform(self, image, matrix):
        """Helper function to apply affine transformation"""
        height, width = image.shape[:2]
        return cv2.warpAffine(image, matrix, (width, height))

    # Previous effect methods remain the same...
    def create_working_effect(self, image, frames=60, intensity=1.0):
        """Create working animation effect (arm movement)"""
        height, width = image.shape[:2]
        result = []
        
        arm_top = height // 3
        arm_bottom = 2 * height // 3
        arm_left = width // 3
        arm_right = 2 * width // 3
        
        for i in range(frames):
            frame = image.copy()
            angle = 10 * intensity * np.sin(2 * np.pi * i / 30)
            
            center = ((arm_right - arm_left) // 2, (arm_bottom - arm_top) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            arm_region = frame[arm_top:arm_bottom, arm_left:arm_right]
            rotated_arm = cv2.warpAffine(arm_region, M, 
                                       (arm_right - arm_left, arm_bottom - arm_top))
            
            frame[arm_top:arm_bottom, arm_left:arm_right] = rotated_arm
            result.append(frame)
        return result

    def create_typing_effect(self, image, frames=60, intensity=1.0):
        """Create typing animation effect"""
        height, width = image.shape[:2]
        result = []
        
        hands_top = 2 * height // 3
        hands_bottom = height
        
        for i in range(frames):
            frame = image.copy()
            vertical_shift = int(5 * intensity * np.sin(4 * np.pi * i / 30))
            
            M = np.float32([[1, 0, 0], [0, 1, vertical_shift]])
            hands_region = frame[hands_top:hands_bottom, :]
            shifted_hands = cv2.warpAffine(hands_region, M, 
                                         (width, hands_bottom - hands_top))
            
            frame[hands_top:hands_bottom, :] = shifted_hands
            result.append(frame)
        return result

    def create_nodding_effect(self, image, frames=60, intensity=1.0):
        """Create nodding animation effect"""
        height, width = image.shape[:2]
        result = []
        
        head_height = height // 3
        
        for i in range(frames):
            frame = image.copy()
            angle = 5 * intensity * np.sin(2 * np.pi * i / 30)
            
            center = (width // 2, head_height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            head_region = frame[:head_height, :]
            rotated_head = cv2.warpAffine(head_region, M, (width, head_height))
            
            frame[:head_height, :] = rotated_head
            result.append(frame)
        return result

def process_image(uploaded_file, target_size=(640, 480)):
    """Process uploaded image with error handling"""
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(image), None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def main():
    st.set_page_config(page_title="Advanced Image Animator", 
                      page_icon="🎬", 
                      layout="wide")
    
    st.title("🎨 Advanced Image Animation Studio")
    
    # Initialize session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'animation_history' not in st.session_state:
        st.session_state.animation_history = []
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Create Animation", "History", "Help"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Upload & Settings")
            uploaded_file = st.file_uploader("Upload Image", 
                                           type=["png", "jpg", "jpeg"])
            
            # Animation settings
            animation_type = st.selectbox("Animation Type",
                                        ["walking", "working", "typing", 
                                         "nodding", "floating", "spinning", 
                                         "pulse"])
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                fps = st.slider("Frames per second", 15, 60, 30)
                duration = st.slider("Duration (seconds)", 1, 10, 3)
                intensity = st.slider("Effect Intensity", 0.1, 2.0, 1.0)
                
                # Image enhancement options
                st.subheader("Image Enhancement")
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
                blur = st.slider("Blur", 0.0, 5.0, 0.0)
                sharpen = st.checkbox("Sharpen")
            
        with col2:
            st.subheader("Preview & Output")
            
            if uploaded_file:
                image, error = process_image(uploaded_file)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.processed_image = image
                    st.image(image, caption="Original Image", 
                            use_column_width=True)
                    
                    if st.button("Generate Animation"):
                        try:
                            animator = AdvancedImageAnimator()
                            
                            # Apply image filters
                            filters = {
                                'brightness': brightness,
                                'contrast': contrast,
                                'blur': blur,
                                'sharpen': sharpen
                            }
                            
                            enhanced_image = animator.apply_image_filters(
                                image, filters)
                            
                            # Create animation frames
                            frames = animator.effects[animation_type](
                                enhanced_image, 
                                frames=duration * fps,
                                intensity=intensity
                            )
                            
                            # Create video
                            with tempfile.TemporaryDirectory() as temp_dir:
                                output_path = os.path.join(temp_dir, 
                                                         "animation.mp4")
                                clip = ImageSequenceClip(frames, fps=fps)
                                clip.write_videofile(output_path, fps=fps, 
                                                   codec='libx264', 
                                                   audio=False)
                                
                                with open(output_path, 'rb') as f:
                                    video_bytes = f.read()
                                
                                # Save to history
                                st.session_state.animation_history.append({
                                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'type': animation_type,
                                    'video': video_bytes
                                })
                                
                                # Display video and download button
                                st.video(video_bytes)
                                st.download_button(
                                    "Download Animation",
                                    video_bytes,
                                    "animation.mp4",
                                    "video/mp4"
                                )
                                
                        except Exception as e:
                            st.error(f"Animation error: {str(e)}")
            else:
                st.info("Please upload an image to start.")
    
    with tab2:
        st.subheader("Animation History")
        if st.session_state.animation_history:
            for idx, anim in enumerate(reversed(
                st.session_state.animation_history)):
                with st.expander(
                    f"Animation {idx+1} - {anim['type']} "
                    f"({anim['timestamp']})"):
                    st.video(anim['video'])
                    st.download_button(
                        f"Download Animation {idx+1}",
                        anim['video'],
                        f"animation_{idx+1}.mp4",
                        "video/mp4"
                    )
        else:
            st.info("No animations created yet.")
    
    with tab3:
        st.subheader("How to Use")
        st.markdown("""
        1. **Upload Image**: Start by uploading an image in JPG or PNG format
        2. **Choose Animation**: Select from various animation effects
        3. **Adjust Settings**: Customize FPS, duration, and intensity
        4. **Enhance Image**: Use advanced settings to adjust brightness, 
           contrast, and other effects
        5. **Generate**: Click 'Generate Animation' to create your animation
        6. **Download**: Save your animation as an MP4 file
        
        **Available Effects:**
        - Walking: Creates walking motion
        - Working: Simulates working/arm movement
        - Typing: Creates typing animation
        - Nodding: Head nodding animation
        - Floating: Hovering effect
        - Spinning: Rotation effect
        - Pulse: Scale pulsing effect
        
        **Tips:**
        - Higher FPS creates smoother animations but takes longer to process
        - Adjust intensity to control the strength of the effect
        - Use image enhancement to improve the final result
        """)

if __name__ == "__main__":
    # Handling import error for OpenCV
    try:
        import cv2
    except ImportError:
        st.error("""
        OpenCV (cv2) is not installed. Please install it using:
        ```
        pip install opencv-python
        ```
        If you're using Streamlit Cloud, make sure to include opencv-python 
        in your requirements.txt file.
        """)
        st.stop()
    
    main()
