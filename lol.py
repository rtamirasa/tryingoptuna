import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import optuna
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the image processing environment
class ImageProcessingEnv(gym.Env):
    def __init__(self, original_image, reference_image):
        super(ImageProcessingEnv, self).__init__()
        self.original_image = original_image
        self.reference_image = reference_image
        self.action_space = spaces.Discrete(3)  # Three actions: brightness, sharpening, denoising
        self.observation_space = spaces.Box(low=0, high=255, shape=original_image.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        self.current_image = self.original_image.copy()
        return self.current_image, {}

    def step(self, action):
        logging.info(f"Optimizing action: {action}")
        if action == 0:
            # Optimize brightness using Optuna in parallel
            study = optuna.create_study(direction='maximize')
            study.optimize(self.brightness_objective, n_trials=10, n_jobs=4)  # Run trials in parallel
            best_brightness = study.best_params['brightness']
            best_contrast = study.best_params['contrast']
            self.current_image = self.adjust_image(self.current_image, best_brightness, best_contrast)
        elif action == 1:
            # Optimize sharpening using Optuna in parallel
            study = optuna.create_study(direction='maximize')
            study.optimize(self.sharpening_objective, n_trials=10, n_jobs=4)
            best_sharpening_strength = study.best_params['sharpening_strength']
            self.current_image = self.apply_sharpening(self.current_image, best_sharpening_strength)
        elif action == 2:
            # Optimize denoising using Optuna in parallel
            study = optuna.create_study(direction='maximize')
            study.optimize(self.denoising_objective, n_trials=10, n_jobs=4)
            best_h = study.best_params['h']
            self.current_image = self.apply_denoising(self.current_image, best_h)

        reward = self.evaluate_image_quality(self.current_image, self.reference_image)
        done = True  # Single-step environment for simplicity
        return self.current_image, reward, done, {}, {}

    def brightness_objective(self, trial):
        brightness = trial.suggest_float('brightness', -50, 50)
        contrast = trial.suggest_float('contrast', 0.8, 1.2)
        img_adjusted = self.adjust_image(self.original_image, brightness, contrast)
        return self.evaluate_image_quality(img_adjusted, self.reference_image)

    def sharpening_objective(self, trial):
        sharpening_strength = trial.suggest_float('sharpening_strength', 0, 5)
        img_sharpened = self.apply_sharpening(self.original_image, sharpening_strength)
        return self.evaluate_image_quality(img_sharpened, self.reference_image)

    def denoising_objective(self, trial):
        h = trial.suggest_float('h', 0, 10)
        img_denoised = self.apply_denoising(self.original_image, h)
        return self.evaluate_image_quality(img_denoised, self.reference_image)

    def adjust_image(self, image, brightness, contrast):
        img_float = image.astype(np.float32)
        img_adjusted = cv2.addWeighted(img_float, contrast, np.zeros_like(img_float), 0, brightness)
        img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)
        return img_adjusted

    def apply_sharpening(self, image, sharpening_strength):
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpening_strength, -1], [-1, -1, -1]])
        img_sharpened = cv2.filter2D(image, -1, kernel)
        return img_sharpened

    def apply_denoising(self, image, h):
        img_denoised = cv2.GaussianBlur(image, (5, 5), h)  # Example Gaussian denoising
        return img_denoised

    def evaluate_image_quality(self, image, reference):
        psnr_value = cv2.PSNR(image, reference)
        return psnr_value

# List of image paths to process
image_paths = ['Images/1004.jpg', 'Images/10.jpg', 'Images/100.jpg']
reference_image_path = 'Images/1041.jpg'

# Load reference image
reference_image = cv2.imread(reference_image_path)
reference_image = cv2.resize(reference_image, (640, 480))

# Process each image in the list
for image_path in image_paths:
    # Load and resize the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (640, 480))

    # Create the environment for the current image
    env = ImageProcessingEnv(original_image, reference_image)

    # Train the agent using PPO
    model = PPO('CnnPolicy', env, verbose=1, batch_size=16, n_steps=512)
    model.learn(total_timesteps=100)

    # Test the trained agent
    obs, _ = env.reset()
    action, _states = model.predict(obs)
    obs, reward, done, info, _ = env.step(action)

    # Save the processed image
    output_path = f'processed_{image_path.split("/")[-1]}'
    cv2.imwrite(output_path, obs)
    print(f'Saved processed image to {output_path}')

    # Optionally display the processed image
    cv2.imshow('Processed Image', obs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
