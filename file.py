import cv2
import os
import datetime
import time
import numpy as np
import pickle
from abc import ABC, abstractmethod
from collections import Counter # Import Counter for mark_attendance

# Directory structure setup
DATA_DIR = "data"
USERS_DIR = os.path.join(DATA_DIR, "users")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")
FACE_SIZE = (200, 200)  # Increased for better resolution

# Ensure directories exist
for directory in [DATA_DIR, USERS_DIR, ATTENDANCE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

class FaceDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        pass

class HaarCascadeDetector(FaceDetector):
    """Face detector using Haar Cascade classifier"""
    
    def __init__(self):
        # Ensure the Haar Cascade XML file is correctly loaded
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.__face_cascade.empty():
            print("Error: Could not load Haar Cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is in the OpenCV data path.")
            exit() # Exit if the classifier can't be loaded

    def detect_faces(self, frame):
        """Detect faces in the given frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray

class FaceRecognizer(ABC):
    """Abstract base class for face recognizers"""
    
    @abstractmethod
    def compare_faces(self, face1, face2):
        """Compare two faces and return similarity score"""
        pass
    
    @abstractmethod
    def identify_user(self, face_img):
        """Identify a user from their face image"""
        pass

class AdvancedFaceRecognizer(FaceRecognizer):
    """Advanced face recognizer using multiple techniques"""
    
    def __init__(self, threshold=0.6):
        self.__threshold = threshold
        self.__sift = cv2.SIFT_create()  # Scale-Invariant Feature Transform
        
    @property
    def threshold(self):
        return self.__threshold
    
    @threshold.setter
    def threshold(self, value):
        if 0.1 <= value <= 0.9:
            self.__threshold = value
        else:
            print("Warning: Threshold must be between 0.1 and 0.9. Value not set.")
    
    def __extract_features(self, face_img):
        """Extract SIFT features from face image"""
        # Ensure image is grayscale
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        # Apply histogram equalization to improve contrast
        face_img = cv2.equalizeHist(face_img)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.__sift.detectAndCompute(face_img, None)
        
        return keypoints, descriptors
    
    def __compute_lbp_histogram(self, face_img):
        """Compute Local Binary Pattern histogram"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        # Apply LBP-like operation manually
        rows, cols = face_img.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = face_img[i, j]
                code = 0
                code |= (face_img[i-1, j-1] >= center) << 7
                code |= (face_img[i-1, j] >= center) << 6
                code |= (face_img[i-1, j+1] >= center) << 5
                code |= (face_img[i, j+1] >= center) << 4
                code |= (face_img[i+1, j+1] >= center) << 3
                code |= (face_img[i+1, j] >= center) << 2
                code |= (face_img[i+1, j-1] >= center) << 1
                code |= (face_img[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def compare_faces(self, face1, face2):
        """
        Compare two face images using multiple metrics
        Returns tuple (match_bool, similarity_score)
        """
        # Ensure both images are the same size
        face1 = cv2.resize(face1, FACE_SIZE)
        face2 = cv2.resize(face2, FACE_SIZE)
        
        # Compute histograms for traditional comparison
        hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms (correlation method)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # LBP histograms comparison
        lbp_hist1 = self.__compute_lbp_histogram(face1)
        lbp_hist2 = self.__compute_lbp_histogram(face2)
        lbp_score = cv2.compareHist(
            np.float32(lbp_hist1), 
            np.float32(lbp_hist2), 
            cv2.HISTCMP_CORREL
        )
        
        # SIFT feature matching (if we have enough keypoints)
        try:
            keypoints1, descriptors1 = self.__extract_features(face1)
            keypoints2, descriptors2 = self.__extract_features(face2)
            
            if descriptors1 is not None and descriptors2 is not None and len(keypoints1) > 10 and len(keypoints2) > 10:
                # Use FLANN based matcher (Fast Library for Approximate Nearest Neighbors)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                sift_score = len(good_matches) / max(len(keypoints1), len(keypoints2))
            else:
                sift_score = 0
                
        except (cv2.error, TypeError, ValueError) as e:
            sift_score = 0
            
        # Structural Similarity Index (SSIM)
        try:
            ssim_score = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)[0][0]
            ssim_score = (ssim_score + 1) / 2  # Convert to 0-1 range
        except Exception as e:
            ssim_score = 0
            
        # Combine scores (weighted average)
        combined_score = (
            0.35 * hist_score + 
            0.35 * lbp_score + 
            0.15 * sift_score + 
            0.15 * ssim_score
        )
        
        return combined_score >= self.__threshold, combined_score
    
    def identify_user(self, face_img):
        """
        Identify a user by comparing with registered faces
        Returns tuple (user_name, similarity_score) or (None, score)
        """
        max_similarity = -1
        identified_user = None
        
        # Loop through all registered users
        for filename in os.listdir(USERS_DIR):
            if filename.endswith('.pkl'):
                user_path = os.path.join(USERS_DIR, filename)
                try:
                    with open(user_path, 'rb') as f:
                        user_data = pickle.load(f)
                except Exception as e:
                    print(f"DEBUG: Error loading user data from {user_path}: {e}")
                    continue 
                    
                username = user_data['name']
                stored_faces = user_data['faces']
                
                user_max_similarity = -1
                
                for stored_face in stored_faces:
                    if not isinstance(stored_face, np.ndarray):
                        # print(f"DEBUG: Warning: Stored face for {username} is not a numpy array. Skipping.")
                        continue
                    
                    match, similarity = self.compare_faces(face_img, stored_face)
                    user_max_similarity = max(user_max_similarity, similarity)
                
                if user_max_similarity > max_similarity:
                    max_similarity = user_max_similarity
                    identified_user = username
        
        print(f"DEBUG: identify_user result - User: {identified_user}, Max Similarity: {max_similarity:.2f}, Threshold: {self.__threshold:.2f}")

        if max_similarity >= self.__threshold:
            return identified_user, max_similarity
        else:
            return None, max_similarity

class FaceEnhancer:
    """Class for applying various face enhancement techniques"""
    def __init__(self):
        pass

    @staticmethod
    def enhance_contrast(face_img):
        """Enhance contrast using histogram equalization"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(face_img)
    
    @staticmethod
    def reduce_noise(face_img):
        """Reduce noise using Gaussian blur"""
        return cv2.GaussianBlur(face_img, (5, 5), 0) 
    
    @staticmethod
    def sharpen(face_img):
        """Sharpen image using unsharp masking"""
        blurred = cv2.GaussianBlur(face_img, (0, 0), 3) 
        return cv2.addWeighted(face_img, 1.5, blurred, -0.5, 0)
    
    @staticmethod
    def adaptive_threshold(face_img):
        """Apply adaptive thresholding"""
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            face_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

class AttendanceSystem:
    """Main class for the Facial Attendance System"""
    
    def __init__(self):
        self.__face_detector = HaarCascadeDetector()
        self.__face_recognizer = AdvancedFaceRecognizer()
        self.__face_enhancer = FaceEnhancer()
        self.__last_marked_user = {} # Dictionary to store last marked time for each user

    def __capture_face(self, num_samples=5, enhance=True):
        """
        Capture multiple face images from webcam
        Returns a list of captured face images
        """
        print(f"Capturing {num_samples} face images - please look at the camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera. Please check camera connection and permissions.")
            return None
        
        time.sleep(1)
        
        face_samples = []
        while len(face_samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            frame = cv2.flip(frame, 1) 
            
            faces, gray = self.__face_detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            samples_text = f"Samples: {len(face_samples)}/{num_samples}"
            cv2.putText(frame, samples_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, "Press 'c' to capture or 'q' to quit", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Face Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                
                if enhance:
                    face_img = self.__face_enhancer.enhance_contrast(face_img)
                    face_img = self.__face_enhancer.reduce_noise(face_img) 
                
                face_samples.append(face_img)
                print(f"Sample {len(face_samples)}/{num_samples} captured")
                
                cv2.imshow('Captured Face', face_img)
                cv2.waitKey(500) 
                
                time.sleep(0.5)
                
            elif key == ord('q'):
                print("Face capture aborted.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) > 0:
            print(f"Captured {len(face_samples)} face samples.")
            return face_samples
        else:
            print("No faces were captured.")
            return None
    
    def register_new_user(self):
        """Register a new user by capturing their face"""
        name = input("Enter your name: ").strip()
        if not name:
            print("Name cannot be empty. Registration cancelled.")
            return
            
        filename = name.lower().replace(" ", "_")
        user_file = os.path.join(USERS_DIR, f"{filename}.pkl")
        
        user_data = {
            'name': name,
            'faces': []
        }

        if os.path.exists(user_file):
            print(f"User {name} already exists. Loading existing data to add more samples.")
            try:
                with open(user_file, 'rb') as f:
                    user_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading existing user data: {e}. Starting with new data.")
        
        face_samples = self.__capture_face(num_samples=5)
        
        if face_samples:
            user_data['faces'].extend(face_samples)
            
            try:
                with open(user_file, 'wb') as f:
                    pickle.dump(user_data, f)
                print(f"User '{name}' registered successfully with {len(face_samples)} new face samples!")
                print(f"Total face samples for '{name}': {len(user_data['faces'])}")
            except Exception as e:
                print(f"Error saving user data for {name}: {e}")
        else:
            print("Registration failed. No faces were captured.")
    
    def mark_attendance(self):
        """Mark attendance for an existing user"""
        print("Looking for a registered face. Please present your face to the camera. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera. Please check camera connection and permissions.")
            return
        
        attendance_marked_for_session = False
        start_time = time.time()
        recognition_buffer = [] # To store recent recognitions
        
        while time.time() - start_time < 30: # Try for 30 seconds max
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            frame = cv2.flip(frame, 1) 
            
            faces, gray = self.__face_detector.detect_faces(frame)
            
            display_text = "Scanning..."
            text_color = (0, 0, 255) # Red for unknown/scanning
            
            if len(faces) > 0:
                print(f"DEBUG: Faces detected: {len(faces)}")
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                
                enhanced_face = self.__face_enhancer.enhance_contrast(face_img)
                enhanced_face = self.__face_enhancer.reduce_noise(enhanced_face)
                
                user, similarity = self.__face_recognizer.identify_user(enhanced_face)
                
                recognition_buffer.append((user, similarity))
                if len(recognition_buffer) > 15: 
                    recognition_buffer.pop(0)
                
                print(f"DEBUG: Current recognition buffer ({len(recognition_buffer)}): {recognition_buffer}")

                stable_user = None
                if len(recognition_buffer) >= 2: 
                    valid_recognitions = [r[0] for r in recognition_buffer if r[0] is not None]
                    
                    if valid_recognitions:
                        user_counts = Counter(valid_recognitions)
                        most_common_user, count = user_counts.most_common(1)[0]
                        
                        if count >= int(len(recognition_buffer) * 0.7): 
                            stable_user = most_common_user
                            avg_similarity = np.mean([s for u, s in recognition_buffer if u == stable_user])
                            user = stable_user 
                            similarity = avg_similarity
                            print(f"DEBUG: Stable user identified: {stable_user} with avg similarity {avg_similarity:.2f}")

                if user:
                    display_text = f"Recognized: {user} ({similarity:.2f})"
                    text_color = (0, 255, 0) # Green for recognized
                    
                    if stable_user and not attendance_marked_for_session:
                        print(f"DEBUG: Checking attendance for stable user: {stable_user}")
                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H:%M:%S")
                        
                        attendance_file = os.path.join(ATTENDANCE_DIR, f"{date_str}.txt")
                        
                        user_already_marked = False
                        if os.path.exists(attendance_file):
                            with open(attendance_file, "r") as f:
                                for line in f:
                                    if line.startswith(stable_user + ","):
                                        user_already_marked = True
                                        print(f"DEBUG: User '{stable_user}' already marked today.")
                                        break
                        
                        if not user_already_marked:
                            if not os.path.exists(attendance_file):
                                with open(attendance_file, "w") as f:
                                    f.write("Name,Time,Confidence\n")
                                print(f"DEBUG: Created new attendance file: {attendance_file}")
                            
                            with open(attendance_file, "a") as f:
                                f.write(f"{stable_user},{time_str},{similarity:.2f}\n")
                            
                            print(f"ATTENDANCE MARKED: '{stable_user}' at {time_str} with confidence {similarity:.2f}")
                            attendance_marked_for_session = True 
                            display_text = f"Attendance Marked: {stable_user}!"
                            text_color = (255, 0, 0) 
                            time.sleep(2) 
                            break 
                        else:
                            display_text = f"{stable_user} already marked attendance today."
                            text_color = (0, 255, 255) 
                            attendance_marked_for_session = True 
                            time.sleep(2) 
                            break 
                else:
                    display_text = "Unknown Face"
                    text_color = (0, 0, 255) 
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), text_color, 2)
                cv2.putText(frame, display_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            else:
                display_text = "No face detected"
                text_color = (0, 0, 255) 

            cv2.putText(frame, f"Threshold: {self.__face_recognizer.threshold:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Mark Attendance', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Attendance marking aborted.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not attendance_marked_for_session:
            print("No registered face was recognized or attendance was not marked within the time limit.")
    
    def view_attendance(self):
        """View attendance records"""
        attendance_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.txt')]
        
        if not attendance_files:
            print("No attendance records found.")
            return
        
        attendance_files.sort(reverse=True) 
        
        print("\n--- Available Attendance Dates ---")
        for i, filename in enumerate(attendance_files):
            date = filename.split('.')[0]
            print(f"{i+1}. {date}")
        print("----------------------------------")
        
        choice = input("Select a date (number) or 'all' to see all records: ").strip()
        
        if choice.lower() == 'all':
            for filename in attendance_files:
                date = filename.split('.')[0]
                print(f"\n--- Attendance for {date} ---")
                try:
                    with open(os.path.join(ATTENDANCE_DIR, filename), "r") as f:
                        lines = f.readlines()
                        if not lines:
                            print("    No records for this date.")
                            continue
                        for line in lines:
                            print(line.strip())
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(attendance_files):
                    filename = attendance_files[index]
                    date = filename.split('.')[0]
                    print(f"\n--- Attendance for {date} ---")
                    with open(os.path.join(ATTENDANCE_DIR, filename), "r") as f:
                        lines = f.readlines()
                        if not lines:
                            print("No records for this date.")
                        for line in lines:
                            print(line.strip())
                else:
                    print("Invalid selection. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number or 'all'.")
    
    def adjust_sensitivity(self):
        """Adjust face recognition sensitivity"""
        print(f"Current recognition threshold: {self.__face_recognizer.threshold:.2f}")
        print("Lower values (e.g., 0.4) make recognition more lenient (easier to match).")
        print("Higher values (e.g., 0.8) make recognition more strict (harder to match).")
        
        try:
            new_threshold_str = input("Enter new threshold (0.1-0.9): ").strip()
            if not new_threshold_str:
                print("No input provided. Threshold remains unchanged.")
                return

            new_threshold = float(new_threshold_str)
            if 0.1 <= new_threshold <= 0.9:
                self.__face_recognizer.threshold = new_threshold
                print(f"Recognition threshold updated to {self.__face_recognizer.threshold:.2f}")
            else:
                print("Invalid value. Threshold must be between 0.1 and 0.9.")
        except ValueError:
            print("Invalid input. Please enter a numerical value.")
    
    def improve_face(self):
        """Face enhancement demo with visualization"""
        print("Capturing a single face for enhancement demonstration...")
        face_samples = self.__capture_face(num_samples=1, enhance=False)
        
        if not face_samples:
            print("No face captured for enhancement demo.")
            return
        
        face_img = face_samples[0]
        
        if len(face_img.shape) > 2:
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            face_img_gray = face_img

        enhanced_contrast = self.__face_enhancer.enhance_contrast(face_img_gray)
        reduced_noise = self.__face_enhancer.reduce_noise(face_img_gray)
        sharpened = self.__face_enhancer.sharpen(face_img_gray)
        adaptive_thresh = self.__face_enhancer.adaptive_threshold(face_img_gray)
        
        display_size = (300, 300) 
        face_img_display = cv2.resize(face_img_gray, display_size)
        enhanced_contrast_display = cv2.resize(enhanced_contrast, display_size)
        reduced_noise_display = cv2.resize(reduced_noise, display_size)
        sharpened_display = cv2.resize(sharpened, display_size)
        adaptive_thresh_display = cv2.resize(adaptive_thresh, display_size) 
        
        def add_label(img, label):
            display_img = img.copy()
            if len(display_img.shape) < 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return display_img

        top_left = add_label(face_img_display, "Original")
        top_right = add_label(enhanced_contrast_display, "Contrast Enhanced")
        top_row_display = np.hstack((top_left, top_right))

        bottom_left = add_label(reduced_noise_display, "Noise Reduced")
        bottom_right = add_label(sharpened_display, "Sharpened")
        bottom_row_display = np.hstack((bottom_left, bottom_right))
        
        comparison_display = np.vstack((top_row_display, bottom_row_display))

        cv2.imshow("Face Enhancements Demo", comparison_display)
        
        print("Displaying face enhancement effects. Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\nFace enhancement techniques demonstrated:")
        print("1. **Histogram Equalization**: Improves contrast, especially in uneven lighting.")
        print("2. **Gaussian Blur**: Reduces image noise, which can help with recognition accuracy.")
        print("3. **Sharpening (Unsharp Masking)**: Enhances edges and details for better feature extraction.")
        print("\nThese techniques are applied automatically to detected faces before recognition.")
    
    def run(self):
        """Run the main menu loop"""
        while True:
            print("\n--- FACIAL ATTENDANCE SYSTEM MENU ---")
            print("1. Mark Attendance (Existing User)")
            print("2. Register New User")
            print("3. View Attendance Records")
            print("4. Adjust Recognition Sensitivity")
            print("5. Demonstrate Face Enhancement")
            print("6. Exit")
            print("-------------------------------------")
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                self.mark_attendance()
            elif choice == '2':
                self.register_new_user()
            elif choice == '3':
                self.view_attendance()
            elif choice == '4':
                self.adjust_sensitivity()
            elif choice == '5':
                self.improve_face()
            elif choice == '6':
                print("Thank you for using the Facial Attendance System! Goodbye.")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()