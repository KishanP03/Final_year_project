import cv2
import numpy as np

class VideoStitcher:
    def __init__(self, detector_type="sift"):
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.H = None  # Homography matrix to be reused

    def compute_homography(self, frame1, frame2):
        """Compute the homography matrix for the first frame pair."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        # Match descriptors
        if self.detector_type == "sift":
            matches = self.bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        elif self.detector_type == "orb":
            good_matches = self.bf.match(des1, des2)
            good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Need at least 4 matches to compute homography
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        else:
            return None

    def process_frame(self, frame1, frame2):
        """Process two frames to stitch them into one panorama frame."""
        if self.H is None:
            self.H = self.compute_homography(frame1, frame2)
            if self.H is None:
                print("Failed to compute homography!")
                return None
        
        # Warp the first image to align with the second image
        height, width, channels = frame2.shape
        panorama = cv2.warpPerspective(frame1, self.H, (width + frame1.shape[1], height))
        panorama[0:height, 0:width] = frame2

        return panorama

def stitch_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Get the properties of the video
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

    stitcher = VideoStitcher(detector_type="sift")

    # Read the first frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error reading video files.")
        return

    # Compute homography for the first frame pair
    stitcher.H = stitcher.compute_homography(frame1, frame2)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Process frame using precomputed homography
        panorama = stitcher.process_frame(frame1, frame2)

        if panorama is not None:
            out.write(panorama)

            # Display the resulting panorama frame
            cv2.imshow('Panorama', panorama)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Not enough matches found!")

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path1 = 'C:/Users/ganes/Documents/Final Year project/wav2.mp4'
    video_path2 = 'C:/Users/ganes/Documents/Final Year project/wav3.mp4'
    output_path = 'newstitched.mp4'

    stitch_videos(video_path1, video_path2, output_path)
