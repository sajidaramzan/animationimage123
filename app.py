import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from moviepy.editor import ImageSequenceClip
from pathlib import Path
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Image to Video Animation",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = None

def create_transition(img1, img2, effect="fade", steps=30):
    """Create transition between two images"""
    frames = []
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    
    if effect == "fade":
        for i in range(steps):
            alpha = i / steps
            frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            frames.append(frame.astype(np.uint8))
    elif effect == "slide":
        for i in range(steps):
            frame = img1.copy()
            offset = int((i/steps) * img1.shape[1])
            frame[:, :offset] = img2[:, :offset]
            frames.append(frame.astype(np.uint8))
    elif effect == "zoom":
        for i in range(steps):
            scale = 1 + (i/steps)
            center_x, center_y = img1.shape[1] // 2, img1.shape[0] // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            frame = cv2.warpAffine(img1.astype(np.uint8), M, (img1.shape[1], img1.shape[0]))
            frames.append(frame)
    
    return frames

def process_image(uploaded_file, target_size=(640, 480)):
    """Process uploaded image to consistent size"""
    # Read image using PIL
    image = Image.open(uploaded_file)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    return np.array(image)

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
                    col.image(processed_images[idx],
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
                    
                    clip = ImageSequenceClip(frames, fps=fps)
                    clip.write_videofile(output_path, fps=fps, codec='libx264', audio=False)
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
    main()
