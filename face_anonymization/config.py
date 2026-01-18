CAMERA_INDEX = 0


DETECTION_METHOD = "mtcnn"   
DETECT_EVERY_N_FRAMES = 5    


USE_RECOGNITION = True
RECOGNITION_METHOD = "arcface"  
AUTH_THRESHOLD = 0.97  # Similarity threshold: Bu değerin üzerindeki benzerlikte olanlar authorized olur


KNOWN_FACES_DIR = "known_faces"
RESULTS_DIR = "results"


ANONYMIZATION_METHOD = "blur"   
BLUR_KERNEL_SIZE = 51           


USE_TRACKING = False


SHOW_FPS = True
DRAW_BOXES = True