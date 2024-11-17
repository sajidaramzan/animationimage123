import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from moviepy.editor import ImageSequenceClip
import random
from pathlib import Path
import time

def create_transition(img1, img2, effect="fade", steps=30):
    """Create transition between two images"""
    if effect == "fade":
        return [cv2.addWeighted(img1, 1 - i/steps, img2, i/steps, 0) 
                for i in range(steps)]
    elif effect == "slide":
        frames = []
        for i in range(steps):
            frame = img1.copy()
            offset = int((i/steps) * img1.shape[1])
            frame[:, :offset] = img2[:, :offset]
            frames.append(frame)
        return frames
    elif effect == "zoom":
        frames = []
        for i in range(steps):
            scale = 1 + (i/steps)
            center_x, center_y = img1.shape[1] // 2, img1.shape[0] // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            frame = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
            frames.append(frame)
        return frames

def process_image(image, target_size=(640, 480)):
    """Process uploaded image to consistent size"""
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = np.array(Image.open(image))
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return cv2.resize(img, target_size)

def create_animation(images, effect, fps, transition_duration):
    """Create animation from list of images"""
    frames = []
    for i in range(len(images)-1):
        # Add current image frames
        frames.append(images[i])
        
        # Add transition frames
        transition_frames = create_transition(
            images[i], 
            images[i+1], 
            effect=effect, 
            steps=int(fps * transition_duration)
        )
        frames.extend(transition_frames)
    
    # Add last image
    frames.append(images[-1])
    return frames

def main():
    st.title("Advanced Image to Video Animation Creator")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    fps = st.sidebar.slider("Frames per second", 15, 60, 30)
    transition_duration = st.sidebar.slider("Transition Duration (seconds)", 0.1, 2.0, 0.5)
    effect = st.sidebar.selectbox(
        "Transition Effect",
        ["fade", "slide", "zoom"]
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Images (minimum 2)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        # Process images
        with st.spinner("Processing images..."):
            processed_images = [process_image(img) for img in uploaded_files]
            
            # Preview section
            st.subheader("Preview Uploaded Images")
            cols = st.columns(min(4, len(processed_images)))
            for idx, col in enumerate(cols):
                if idx < len(processed_images):
                    col.image(cv2.cvtColor(processed_images[idx], cv2.COLOR_BGR2RGB),
                            caption=f"Image {idx+1}",
                            use_column_width=True)
        
        if st.button("Generate Animation"):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create animation frames
                status_text.text("Creating animation frames...")
                frames = create_animation(
                    processed_images,
                    effect,
                    fps,
                    transition_duration
                )
                progress_bar.progress(50)
                
                # Create temporary directory for video
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save video
                    status_text.text("Generating video file...")
                    output_path = os.path.join(temp_dir, "animation.mp4")
                    
                    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) 
                                           for f in frames], fps=fps)
                    clip.write_videofile(output_path, fps=fps, codec='libx264')
                    progress_bar.progress(90)
                    
                    # Prepare download button
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
        st.info("Please upload at least 2 images to create an animation.")
    
    # Add helpful information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### How to use:
    1. Upload 2 or more images
    2. Adjust settings in sidebar
    3. Click 'Generate Animation'
    4. Download the result
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Image to Video Animation",
        page_icon="🎬",
        layout="wide"
    )
    main()
