import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from moviepy.editor import ImageSequenceClip
from skimage import transform
import io

class ImageAnimator:
    def __init__(self):
        self.effects = {
            'walking': self.create_walking_effect,
            'working': self.create_working_effect,
            'typing': self.create_typing_effect,
            'nodding': self.create_nodding_effect
        }

    def create_walking_effect(self, image, frames=60):
        """Create walking animation effect"""
        height, width = image.shape[:2]
        result = []
        
        # Split image into upper and lower body (approximate)
        upper = image[:height//2, :]
        lower = image[height//2:, :]
        
        for i in range(frames):
            frame = image.copy()
            # Move lower body side to side
            shift = int(10 * np.sin(2 * np.pi * i / 30))
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            shifted_lower = cv2.warpAffine(lower, M, (width, height//2))
            
            # Combine upper and shifted lower body
            frame = image.copy()
            frame[height//2:, :] = shifted_lower
            
            # Add subtle up-down movement
            vertical_shift = int(5 * np.sin(2 * np.pi * i / 15))
            M = np.float32([[1, 0, 0], [0, 1, vertical_shift]])
            frame = cv2.warpAffine(frame, M, (width, height))
            
            result.append(frame)
        return result

    def create_working_effect(self, image, frames=60):
        """Create working animation effect (arm movement)"""
        height, width = image.shape[:2]
        result = []
        
        # Define arm region (approximate)
        arm_top = height // 3
        arm_bottom = 2 * height // 3
        arm_left = width // 3
        arm_right = 2 * width // 3
        
        for i in range(frames):
            frame = image.copy()
            # Create arm movement
            angle = 10 * np.sin(2 * np.pi * i / 30)  # Oscillate between -10 and 10 degrees
            
            # Extract and rotate arm region
            arm_region = frame[arm_top:arm_bottom, arm_left:arm_right]
            center = ((arm_right - arm_left) // 2, (arm_bottom - arm_top) // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_arm = cv2.warpAffine(arm_region, M, (arm_right - arm_left, arm_bottom - arm_top))
            
            # Place rotated arm back
            frame[arm_top:arm_bottom, arm_left:arm_right] = rotated_arm
            result.append(frame)
        return result

    def create_typing_effect(self, image, frames=60):
        """Create typing animation effect"""
        height, width = image.shape[:2]
        result = []
        
        # Define hands region (approximate)
        hands_top = 2 * height // 3
        hands_bottom = height
        
        for i in range(frames):
            frame = image.copy()
            # Create typing movement
            vertical_shift = int(5 * np.sin(4 * np.pi * i / 30))  # Faster movement
            
            # Move hands region
            hands_region = frame[hands_top:hands_bottom, :]
            M = np.float32([[1, 0, 0], [0, 1, vertical_shift]])
            shifted_hands = cv2.warpAffine(hands_region, M, (width, hands_bottom - hands_top))
            
            # Place shifted hands back
            frame_with_hands = frame.copy()
            frame_with_hands[hands_top:hands_bottom, :] = shifted_hands
            result.append(frame_with_hands)
        return result

    def create_nodding_effect(self, image, frames=60):
        """Create nodding animation effect"""
        height, width = image.shape[:2]
        result = []
        
        # Define head region (approximate)
        head_height = height // 3
        
        for i in range(frames):
            frame = image.copy()
            # Create nodding movement
            angle = 5 * np.sin(2 * np.pi * i / 30)  # Oscillate between -5 and 5 degrees
            
            # Extract and rotate head region
            head_region = frame[:head_height, :]
            center = (width // 2, head_height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_head = cv2.warpAffine(head_region, M, (width, head_height))
            
            # Place rotated head back
            frame[:head_height, :] = rotated_head
            result.append(frame)
        return result

    def add_background_motion(self, frames, motion_type="scroll"):
        """Add background motion effect"""
        result = []
        if len(frames) == 0:
            return result
        
        height, width = frames[0].shape[:2]
        
        for i, frame in enumerate(frames):
            if motion_type == "scroll":
                shift = int((i / len(frames)) * width / 4)
                M = np.float32([[1, 0, -shift], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (width, height))
            elif motion_type == "zoom":
                scale = 1 + 0.1 * np.sin(2 * np.pi * i / len(frames))
                M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
                frame = cv2.warpAffine(frame, M, (width, height))
            result.append(frame)
        
        return result

def process_image(uploaded_file, target_size=(640, 480)):
    """Process uploaded image"""
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)

def main():
    st.title("Advanced Image Animation Creator")
    
    animator = ImageAnimator()
    
    # Sidebar controls
    st.sidebar.header("Animation Settings")
    
    animation_type = st.sidebar.selectbox(
        "Animation Type",
        ["walking", "working", "typing", "nodding"]
    )
    
    background_motion = st.sidebar.selectbox(
        "Background Motion",
        ["none", "scroll", "zoom"]
    )
    
    fps = st.sidebar.slider("Frames per second", 15, 60, 30)
    duration = st.sidebar.slider("Duration (seconds)", 1, 10, 3)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        with st.spinner("Processing image..."):
            image = process_image(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Generate Animation"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Create base animation
                    status_text.text("Creating animation...")
                    frames = animator.effects[animation_type](
                        image, 
                        frames=duration * fps
                    )
                    progress_bar.progress(50)
                    
                    # Add background motion if selected
                    if background_motion != "none":
                        frames = animator.add_background_motion(
                            frames, 
                            motion_type=background_motion
                        )
                    progress_bar.progress(75)
                    
                    # Create video
                    with tempfile.TemporaryDirectory() as temp_dir:
                        output_path = os.path.join(temp_dir, "animation.mp4")
                        clip = ImageSequenceClip(frames, fps=fps)
                        clip.write_videofile(output_path, fps=fps, codec='libx264', audio=False)
                        
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        progress_bar.progress(100)
                        status_text.text("Animation complete!")
                        
                        # Display video and download button
                        st.video(video_bytes)
                        st.download_button(
                            label="Download Animation",
                            data=video_bytes,
                            file_name="animation.mp4",
                            mime="video/mp4"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    else:
        st.info("Please upload an image to start.")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Animation Types:
    - Walking: Creates walking motion
    - Working: Simulates working/arm movement
    - Typing: Creates typing animation
    - Nodding: Head nodding animation
    
    ### Background Effects:
    - Scroll: Moving background
    - Zoom: Breathing effect
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Image Animation",
        page_icon="🎬",
        layout="wide"
    )
    main()
