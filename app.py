import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from moviepy.editor import ImageSequenceClip, VideoFileClip, CompositeVideoClip, ColorClip
import mediapipe as mp
from scipy.interpolate import interp1d
import dlib
from rembg import remove
import io

# Initialize face and pose detection
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class ImageAnimator:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def remove_background(self, image):
        """Remove background from image"""
        return remove(image)

    def detect_face_landmarks(self, image):
        """Detect facial landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)

    def detect_pose_landmarks(self, image):
        """Detect body pose landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)

    def create_movement_animation(self, image, movement_type="wave", duration_seconds=3, fps=30):
        """Create movement animation based on detected landmarks"""
        frames = []
        total_frames = duration_seconds * fps
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        if movement_type == "wave":
            # Create wave movement
            for i in range(total_frames):
                frame = image.copy()
                offset = np.sin(2 * np.pi * i / total_frames) * 20
                M = np.float32([[1, 0, offset], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (width, height))
                frames.append(frame)
                
        elif movement_type == "breathe":
            # Create breathing effect
            for i in range(total_frames):
                scale = 1 + 0.05 * np.sin(2 * np.pi * i / total_frames)
                M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
                frame = cv2.warpAffine(image, M, (width, height))
                frames.append(frame)
                
        elif movement_type == "blink":
            # Create blinking effect using facial landmarks
            landmarks = self.detect_face_landmarks(image)
            if len(landmarks) > 0:
                eye_indices = [33, 159, 133, 145]  # Eye landmarks
                for i in range(total_frames):
                    frame = image.copy()
                    if i % 50 < 5:  # Blink every 50 frames
                        # Modify eye region
                        for idx in eye_indices:
                            y = int(landmarks[idx][1] * height)
                            x = int(landmarks[idx][0] * width)
                            cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)
                    frames.append(frame)
            else:
                frames = [image] * total_frames
                
        return frames

def process_image(uploaded_file, target_size=(640, 480)):
    """Process uploaded image"""
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)

def create_composite_animation(frames, background_frames=None, overlay_type="blend"):
    """Create composite animation with background"""
    if background_frames is None:
        return frames
    
    composite_frames = []
    for i in range(len(frames)):
        if overlay_type == "blend":
            alpha = 0.7
            composite = cv2.addWeighted(
                frames[i], 
                alpha, 
                background_frames[i % len(background_frames)], 
                1-alpha, 
                0
            )
        else:  # overlay
            # Add proper alpha blending here
            composite = frames[i]
        
        composite_frames.append(composite)
    
    return composite_frames

def add_special_effects(frames, effect_type="particles"):
    """Add special effects to frames"""
    effect_frames = []
    height, width = frames[0].shape[:2]
    
    if effect_type == "particles":
        particles = np.random.rand(20, 2) * [width, height]
        velocities = np.random.rand(20, 2) * 2 - 1
        
        for frame in frames:
            effect_frame = frame.copy()
            particles += velocities
            
            # Reset particles that go out of bounds
            particles[particles[:, 0] > width, 0] = 0
            particles[particles[:, 1] > height, 1] = 0
            
            # Draw particles
            for particle in particles.astype(int):
                cv2.circle(effect_frame, tuple(particle), 2, (255, 255, 255), -1)
            
            effect_frames.append(effect_frame)
            
    return effect_frames

def main():
    st.title("Advanced Image Animation Creator")
    
    animator = ImageAnimator()
    
    # Sidebar controls
    st.sidebar.header("Animation Settings")
    
    animation_type = st.sidebar.selectbox(
        "Animation Type",
        ["wave", "breathe", "blink", "custom"]
    )
    
    fps = st.sidebar.slider("Frames per second", 15, 60, 30)
    duration = st.sidebar.slider("Duration (seconds)", 1, 10, 3)
    
    # Advanced options
    st.sidebar.header("Advanced Options")
    remove_bg = st.sidebar.checkbox("Remove Background")
    add_effects = st.sidebar.checkbox("Add Particle Effects")
    
    # File uploaders
    main_image = st.file_uploader(
        "Upload Main Image", 
        type=["png", "jpg", "jpeg"]
    )
    
    background_image = st.file_uploader(
        "Upload Background Image (Optional)", 
        type=["png", "jpg", "jpeg"]
    )
    
    if main_image:
        with st.spinner("Processing image..."):
            # Process main image
            image = process_image(main_image)
            
            if remove_bg:
                image = animator.remove_background(image)
            
            st.image(image, caption="Processed Image", use_column_width=True)
            
            if st.button("Generate Animation"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Create animation frames
                    status_text.text("Creating animation frames...")
                    frames = animator.create_movement_animation(
                        image,
                        movement_type=animation_type,
                        duration_seconds=duration,
                        fps=fps
                    )
                    progress_bar.progress(30)
                    
                    # Process background if provided
                    if background_image:
                        background = process_image(background_image)
                        background_frames = [background] * len(frames)
                        frames = create_composite_animation(frames, background_frames)
                    progress_bar.progress(60)
                    
                    # Add special effects if selected
                    if add_effects:
                        frames = add_special_effects(frames)
                    progress_bar.progress(80)
                    
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
        
    # Add instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Features:
    - Multiple animation types
    - Background removal
    - Particle effects
    - Custom duration and FPS
    - Background image support
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Advanced Image Animation",
        page_icon="🎬",
        layout="wide"
    )
    main()
