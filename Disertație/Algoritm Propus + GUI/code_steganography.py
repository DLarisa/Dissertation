import os
import shutil
import cv2
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image, ImageFile
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import pywt
import wave
from scipy.fftpack import dct, idct



workingDir = 'frames'
frame_count = 0
width = 0
height = 0
output_audio_path = 'frames/reconstructed.wav'





###############################################################################
# AES encryption / decryption; binary transformations
def aes_encrypt(plaintext, password):
    key = get_random_bytes(16)  # Generate a random 128-bit key
    cipher = AES.new(password.encode(), AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return key.hex() + ciphertext.hex()

import time
def aes_decrypt(ciphertext, password):
    # key = bytes.fromhex(ciphertext[:32])  # Extract the key from the ciphertext
    error = 0
    try:
        ciphertext = bytes.fromhex(ciphertext[32:])
    except ValueError:
        time.sleep(30)
        error = 1
        return 0, error
    cipher = AES.new(password.encode(), AES.MODE_ECB)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return plaintext.decode(), error


def text_to_binary(text):
    binary_array = []
    for char in text:
        binary = bin(ord(char))[2:].zfill(8)
        binary_array.extend([int(bit) for bit in binary])
    return binary_array

def binary_to_text(binary_array):
    text = ""
    for i in range(0, len(binary_array), 8):
        binary = binary_array[i:i+8]
        binary_str = ''.join(str(bit) for bit in binary)
        decimal = int(binary_str, 2)
        text += chr(decimal)
    return text



###############################################################################
### RGB to YUV
def rgb2yuv(image_name):
    # Load the RGB image
    frame_path = os.path.join(workingDir, 'framesRGB', image_name)
    rgb_image = cv2.imread(frame_path)
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)

    # Split YUV channels
    Y = yuv_image[:, :, 0]
    U = yuv_image[:, :, 1]
    V = yuv_image[:, :, 2]

    # Save each channel as a separate image
    frame_path = os.path.join(workingDir, 'framesY', image_name)
    cv2.imwrite(frame_path, Y)
    frame_path = os.path.join(workingDir, 'framesU', image_name)
    cv2.imwrite(frame_path, U)
    frame_path = os.path.join(workingDir, 'framesV', image_name)
    cv2.imwrite(frame_path, V)



### YUV to RGB
def yuv2rgb(image_name):
    # Load the 3 separate images
    frame_path = os.path.join(workingDir, 'framesY', image_name)
    Y = cv2.imread(frame_path, 0)
    frame_path = os.path.join(workingDir, 'framesU', image_name)
    U = cv2.imread(frame_path, 0)
    frame_path = os.path.join(workingDir, 'framesV', image_name)
    V = cv2.imread(frame_path, 0)

    yuv_image = np.stack((Y, U, V), axis=-1)

    # Convert YUV image to RGB
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YCrCb2RGB)

    # Save the reversed RGB image in the apropiate folder
    frame_path = os.path.join(workingDir, 'framesYUV2RGB', image_name)
    cv2.imwrite(frame_path, rgb_image)
    
    

###############################################################################
### Shuffle Pixels of Image
def shuffleImage(image, seed):
    # Convert image to NumPy array
    pix = np.array(image)

    # Generate an array of shuffled indices
    # Seed random number generation to ensure the same result
    np.random.seed(seed)
    indices = np.random.permutation(len(pix))

    # Shuffle the pixels using the indices
    shuffled = pix[indices].astype(np.uint8)

    # Reshape the shuffled array into the original image dimensions
    shuffled_image = shuffled.reshape(image.shape)

    return shuffled_image



### Unshuffle Pixels of Image
def unshuffleImage(image, seed):
    # Get shuffled pixels in NumPy array
    shuffled = np.array(image)
    nPix = len(shuffled)

    # Generate unshuffler
    np.random.seed(seed)
    indices = np.random.permutation(nPix)
    unshuffler = np.zeros(nPix, np.uint32)
    unshuffler[indices] = np.arange(nPix)

    # Unshuffle the pixels using the unshuffler array
    unshuffledPix = shuffled[unshuffler].astype(np.uint8)

    # Reshape the unshuffled pixels into the original image dimensions
    shuffled_image = unshuffledPix.reshape(image.shape)

    return shuffled_image



###############################################################################
### Create directories, split into frames, turn to YUV and shuffle pixels
def process_video(video_file_path, key1):
    # General Directory for all the Frames
    global workingDir
    
    # If it already exists, delete it in order to avoid conflicts
    if os.path.exists(workingDir):
        shutil.rmtree(workingDir)
        
    # Create the General Directory
    os.makedirs(workingDir)
    # Create the Directory for the video Frames
    os.makedirs(os.path.join(workingDir, 'framesRGB'))
    # Create the Directory for Y, U, V frames
    os.makedirs(os.path.join(workingDir, 'framesY'))
    os.makedirs(os.path.join(workingDir, 'framesU'))
    os.makedirs(os.path.join(workingDir, 'framesV'))
    # Create the Directory for the RGB Frames after conversion from YUV
    os.makedirs(os.path.join(workingDir, 'framesYUV2RGB'))
    
    # Read video and turn it into frames
    video = cv2.VideoCapture(video_file_path)
    global frame_count
    
    # Read and save each frame from the video
    while True:
        # Read the next frame
        ret, frame = video.read()
    
        # Check if a frame was retrieved
        if not ret:
            break
    
        # Save the frame as an image
        name_frame = f'{frame_count}.png'
        frame_path = os.path.join(workingDir, 'framesRGB', name_frame)
        cv2.imwrite(frame_path, frame)
    
        # Increment the frame count
        frame_count += 1
        
    
    # Release the video file
    video.release()
    print(f"Total frames extracted: {frame_count}")
    
    # Take each frame and tranform it into 3 separate photos (Y, U, V)
    for i in range(frame_count):
        # Name of the individual frame (1.png)
        image_name = str(i) + '.png'
        rgb2yuv(image_name)
    
    
    # Scramble the pixels inside the Y, U, V frames using key1
    for i in range(frame_count):
        img_name = str(i) + '.png'
        image_pathY = os.path.join(workingDir, 'framesY', img_name)
        image_pathU = os.path.join(workingDir, 'framesU', img_name)
        image_pathV = os.path.join(workingDir, 'framesV', img_name)
    
        img_y = cv2.imread(image_pathY)
        img_u = cv2.imread(image_pathU)
        img_v = cv2.imread(image_pathV)
    
        result_y = shuffleImage(img_y, key1)
        cv2.imwrite(image_pathY, result_y)
        result_u = shuffleImage(img_u, key1)
        cv2.imwrite(image_pathU, result_u)
        result_v = shuffleImage(img_v, key1)
        cv2.imwrite(image_pathV, result_v)



### Take the photo, turn it into binary and make it an array
def process_image(photo_file_path, key2):
    global width
    global height
    
    #####
    img = cv2.imread(photo_file_path, 2)
    height, width = img.shape
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("photo_aux.png", bw_img)
    #####
    
    # Shuffle the photo and obtain the new one (using key2)
    img = cv2.imread(photo_file_path)
    result = shuffleImage(img, key2)
    cv2.imwrite("hiddenShuffled.png", result)
    
    # Read the image file and turn it to binary form
    img = cv2.imread('hiddenShuffled.png', 2)
    height, width = img.shape
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("hiddenShuffled.png", bw_img)
    
    # Make it 1D array 
    image_test = np.array(Image.open('hiddenShuffled.png'))
    # Manipulate the array
    x = np.array(image_test)
    # Convert to 1D array
    y = np.concatenate(x)
    return y



### Get the Hamming Code from the 1D Array (each 4 bites turn into one sequence of code)
def hamming_process(key1, key2, y):
    # Generator Matrix G (required for encoding)
    G = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1]
    ]
    
    # To get the dimension of one frame
    global workingDir
    frame_path = os.path.join(workingDir, 'framesY', '0.png')
    Y = cv2.imread(frame_path)
    height, width, channels = Y.shape  # we need width
    dimension = height * width - 1
    
    # To see how many Pixels need to be modified in each frame
    # Last frame can have a few more pixels
    global frame_count
    nr_pixel = 0  # to count modified pixels in each frame
    nr_frame = 0  # when reach the last frame we add the remaining pixels; also to move the frames
    pixels_in_frame = (len(y) // 4) // frame_count
    pixels_for_last_frame = (len(y) // 4) % frame_count
    control = 0 # for when we reach last frame and we need to put pixels in
    
    # (Key1 ^ Key2) % 2 == 0 -> beginning, then end; otherwise, reverse 
    parity = 0 # 1 -> begin; -1 -> end 
    if (key1 ^ key2) % 2 == 0:
        parity = 1
    else:
        parity = -1
    
    
    # Open Y, U, V first frames
    name = '0.png'
    path_y = os.path.join(workingDir, 'framesY', name)
    Y = cv2.imread(path_y)
    path_u = os.path.join(workingDir, 'framesU', name)
    U = cv2.imread(path_u)
    path_v = os.path.join(workingDir, 'framesV', name)
    V = cv2.imread(path_v)
    
    
    # From 1D array take group of 4 bits, in order to transform into Hamming Code
    for i in range(0, len(y), 4):
        if i + 4 > len(y):
          break
    
        # Get 4 bits
        aux = y[i:i+4]
        # Multiplication between 2 matrixes (1 matrix with 1 row and 7 columns -> 7 bits)
        aux = np.matmul(aux, G)
        
        # Hamming Code in aux
        for j in range(7):
            # We work in %2 field
            aux[j] %= 2
    
    
        # 3-bits in Y, 2-bits in U (first 2 -> R, G), 2-bits in V (first 2 -> R, G) -> (using LSB)
        if nr_pixel == pixels_in_frame:
            nr_pixel = 0
            nr_frame += 1
    
            # Close the previous YUV frames
            name = str(nr_frame - 1) + '.png'
            path_y = os.path.join(workingDir, 'framesY', name)
            cv2.imwrite(path_y, Y)
            path_u = os.path.join(workingDir, 'framesU', name)
            cv2.imwrite(path_u, U)
            path_v = os.path.join(workingDir, 'framesV', name)
            cv2.imwrite(path_v, V)
            
            # Open Y, U, V new frames
            if nr_frame != frame_count:
                name = str(nr_frame) + '.png'
                path_y = os.path.join(workingDir, 'framesY', name)
                Y = cv2.imread(path_y)
                path_u = os.path.join(workingDir, 'framesU', name)
                U = cv2.imread(path_u)
                path_v = os.path.join(workingDir, 'framesV', name)
                V = cv2.imread(path_v)
            else:
                break
    
    
        # If last frame, we have more pixels (dacă împărțire la rest)
        if control == 0 and nr_frame == frame_count - 1:
            pixels_in_frame += pixels_for_last_frame
            control = 1
    
        # LSB becomes the Hamming Code (stored in aux)
        if parity == 1:
            Y[nr_pixel // width][nr_pixel % width][0] = (Y[nr_pixel // width][nr_pixel % width][0] & ~1) | aux[0]
            Y[nr_pixel // width][nr_pixel % width][1] = (Y[nr_pixel // width][nr_pixel % width][1] & ~1) | aux[1]
            Y[nr_pixel // width][nr_pixel % width][2] = (Y[nr_pixel // width][nr_pixel % width][2] & ~1) | aux[2]
    
            U[nr_pixel // width][nr_pixel % width][0] = (U[nr_pixel // width][nr_pixel % width][0] & ~1) | aux[3]
            U[nr_pixel // width][nr_pixel % width][1] = (U[nr_pixel // width][nr_pixel % width][1] & ~1) | aux[4]
    
            V[nr_pixel // width][nr_pixel % width][0] = (V[nr_pixel // width][nr_pixel % width][0] & ~1) | aux[5]
            V[nr_pixel // width][nr_pixel % width][1] = (V[nr_pixel // width][nr_pixel % width][1] & ~1) | aux[6]
    
            parity = -1
        else:
            contor = dimension - nr_pixel
            Y[contor // width][contor % width][0] = (Y[contor // width][contor % width][0] & ~1) | aux[0]
            Y[contor // width][contor % width][1] = (Y[contor // width][contor % width][1] & ~1) | aux[1]
            Y[contor // width][contor % width][2] = (Y[contor // width][contor % width][2] & ~1) | aux[2]
    
            U[contor // width][contor % width][0] = (U[contor // width][contor % width][0] & ~1) | aux[3]
            U[contor // width][contor % width][1] = (U[contor // width][contor % width][1] & ~1) | aux[4]
    
            V[contor // width][contor % width][0] = (V[contor // width][contor % width][0] & ~1) | aux[5]
            V[contor // width][contor % width][1] = (V[contor // width][contor % width][1] & ~1) | aux[6]
    
            parity = 1
    
        nr_pixel += 1
    
    
    name = str(frame_count - 1) + '.png'
    path_y = os.path.join(workingDir, 'framesY', name)
    cv2.imwrite(path_y, Y)
    path_u = os.path.join(workingDir, 'framesU', name)
    cv2.imwrite(path_u, U)
    path_v = os.path.join(workingDir, 'framesV', name)
    cv2.imwrite(path_v, V)



### Take the embedded frames, unshuffle them, turn from YUV to RGB, turn back to video (.avi)
def transform_back_to_video(key1):
    global workingDir
    global frame_count
    
    # Unshuffle
    for i in range(frame_count):
        img_name = str(i) + '.png'
        image_pathY = os.path.join(workingDir, 'framesY', img_name)
        image_pathU = os.path.join(workingDir, 'framesU', img_name)
        image_pathV = os.path.join(workingDir, 'framesV', img_name)
    
        img_y = cv2.imread(image_pathY)
        img_u = cv2.imread(image_pathU)
        img_v = cv2.imread(image_pathV)
    
        result_y = unshuffleImage(img_y, key1)
        cv2.imwrite(image_pathY, result_y)
        result_u = unshuffleImage(img_u, key1)
        cv2.imwrite(image_pathU, result_u)
        result_v = unshuffleImage(img_v, key1)
        cv2.imwrite(image_pathV, result_v)
        
    
    # Reverse situation (from YUV make back the RGB frames) ---> folder framesYUV2RGB
    for i in range(frame_count):
        image_name = str(i) + '.png'
        yuv2rgb(image_name)
    
    
    # Turn to video
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image_files = []

    for img_number in range(frame_count): 
        image_files.append('frames/framesYUV2RGB/' + str(img_number) + '.png') 
    
    fps = 30
    
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("intermediar.avi", codec='libx264')
   
    
    

    
############################################################################### 
### Enconde and Decode Audio
def encode_audio(data, message, delta = 5):
    segments = np.array_split(data, len(message))
    segments = segments.copy()
    rsegments = []

    for ind, segment in enumerate(segments):

        cA1, cD1 = pywt.dwt(segment, 'db1')

        v = dct(cA1, norm='ortho')

        v1 = v[::2]
        v2 = v[1::2]

        nrmv1 = np.linalg.norm(v1, ord=2)
        nrmv2 = np.linalg.norm(v2, ord=2)

        u1 = v1 / nrmv1
        u2 = v2 / nrmv2

        watermark_bit = message[ind]
        nrm = (nrmv1 + nrmv2) / 2
        if watermark_bit == 1:
            nrmv1 = nrm + delta
            nrmv2 = nrm - delta
        else:
            nrmv1 = nrm - delta
            nrmv2 = nrm + delta

        rv1 = nrmv1 * u1
        rv2 = nrmv2 * u2

        rv = np.zeros((len(v),))

        rv[::2] = rv1
        rv[1::2] = rv2

        rcA1 = idct(rv, norm='ortho')

        rseg = pywt.idwt(rcA1, cD1, 'db1')
        rsegments.append(rseg[:])

    return np.concatenate(rsegments)



def decode_audio(data_with_watermark, watermark_length, delta = 5):
    segments = np.array_split(data_with_watermark, watermark_length)
    segments = segments.copy()
    watermark_bits = []

    for ind, segment in enumerate(segments):
        cA1, cD1 = pywt.dwt(segment, 'db1')

        v = dct(cA1, norm='ortho')

        v1 = v[::2]
        v2 = v[1::2]

        nrmv1 = np.linalg.norm(v1, ord=2)
        nrmv2 = np.linalg.norm(v2, ord=2)

        if nrmv1 > nrmv2:
            watermark_bits.append(1)
        else:
            watermark_bits.append(0)

    return watermark_bits



### Reconstruct Audio from Array
def reconstruct_audio(signal, parameters):
    '''
    Parameters:
      signal - 1D numpy array representing the watermarked signal
      parameters - Tuple of audio parameters obtained from audio.getparams()
    Returns:
      Reconstructed audio signal - 1D numpy array
    '''

    # frames = len(signal)
    sample_width = parameters.sampwidth
    audio_frames = signal.astype(np.uint8 if sample_width == 1 else np.int16).tobytes()

    audio = wave.open("reconstructed.wav", 'wb')
    audio.setparams(parameters)
    audio.writeframes(audio_frames)
    audio.close()
    
    

###############################################################################
### Process audio
def process_audio(video_file_path, key1, key2, key3, y):
    global width
    global height
    
    # Get original audio from original video
    source_video = VideoFileClip(video_file_path)
    source_video.audio.write_audiofile(r"audio_original.wav") 
    
    audio = wave.open("audio_original.wav", 'r')
    parameters = audio.getparams()  # will be used for reconstructing the stego audio
    frames = audio.getnframes() # number of audio frames
    sample_width = audio.getsampwidth() # sample width in bytes
    audio_frames = audio.readframes(frames)
    rawdata = np.frombuffer(audio_frames, dtype=np.uint8 if sample_width == 1 else np.int16)
    rawData = np.copy(rawdata)
    
    # Create the secret message
    message = f'Key1 = {key1}; Key2 = {key2}; Width X Height = {width} X {height}; Len(y) = {y}'
    print(message)
    encrypted_msg = aes_encrypt(message, key3)
    binary_msg = text_to_binary(encrypted_msg)
    print("Length Hidden Data: ", len(binary_msg))
    
    # Encode and create the new stego audio
    aux = encode_audio(rawData, binary_msg)
    reconstruct_audio(aux, parameters)
    
    
    
### Attach stego audio to stego video (create the final video)
def attach_stego_audio(output_path):
   audio_clip = AudioFileClip(r"reconstructed.wav")
   destination_video = VideoFileClip("intermediar.avi")
   final_clip = destination_video.set_audio(audio_clip)
   # save the final clip
   final_clip.write_videofile(output_path, codec='libx264')     
    
    
    
###############################################################################    
### Encode Main Function (Video Stegano)
def encode(video_file_path, photo_file_path, key1, key2, key3, output_path):
    process_video(video_file_path, key1)
    y = process_image(photo_file_path, key2) # len_img
    hamming_process(key1, key2, y)
    transform_back_to_video(key1)
    process_audio(video_file_path, key1, key2, key3, len(y))
    attach_stego_audio(output_path)
    
    
    
###############################################################################
### Function that takes the stego audio and obtains the hidden data
def process_stego_audio(video_file_path, key3, len_data, output_path):
    global output_audio_path
    video_clip = VideoFileClip(video_file_path)
    video_clip.audio.write_audiofile(output_audio_path)
    audio = wave.open("reconstructed.wav", 'r')
    frames = audio.getnframes() # number of audio frames
    sample_width = audio.getsampwidth() # sample width in bytes
    audio_frames = audio.readframes(frames)
    rawdata = np.frombuffer(audio_frames, dtype=np.uint8 if sample_width == 1 else np.int16)
    rawData = np.copy(rawdata)
    result = decode_audio(rawData, len_data)
    newString = binary_to_text(result)
    decrypted_msg, error = aes_decrypt(newString, key3)
    if error:
        img = cv2.imread("photo_aux.png")
        copied_img = img.copy()
        cv2.imwrite(output_path, copied_img)
        return -1, -1, -1, -1, -1
    os.remove(output_audio_path)
    pairs = decrypted_msg.split(";")
    key1 = None
    key2 = None
    width = None
    height = None
    len_y = None
    for pair in pairs:
        pair = pair.strip()
        key, value = pair.split("=")
        key = key.strip()
        value = value.strip()
        if key == "Key1":
            key1 = int(value)
        elif key == "Key2":
            key2 = int(value)
        elif key == "Width X Height":
            width, height = map(int, value.split("X"))
        elif key == "Len(y)":
            len_y = int(value)
    return key1, key2, width, height, len_y



### Extract the hidden photo from the stego video
def extract_hidden_photo(video_file_path, key1, key2, width_photo, height_photo, len_y, output_path):
    global workingDir
    global frame_count
    
    video = cv2.VideoCapture(video_file_path)
    # Check if video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        exit()
    
    frame_count = 0
    # Read and save each frame from the video
    while True:
        # Read the next frame
        ret, frame = video.read()
    
        # Check if a frame was retrieved
        if not ret:
            break
    
        # Save the frame as an image
        name_frame = f'{frame_count}.png'
        frame_path = os.path.join(workingDir, 'framesRGB', name_frame)
        cv2.imwrite(frame_path, frame)
    
        # Increment the frame count
        frame_count += 1
    
    # Release the video file
    video.release()
    print(f"Total frames extracted: {frame_count}")
    
    # Shuffle
    for i in range(frame_count):
        img_name = str(i) + '.png'
        image_pathY = os.path.join(workingDir, 'framesY', img_name)
        image_pathU = os.path.join(workingDir, 'framesU', img_name)
        image_pathV = os.path.join(workingDir, 'framesV', img_name)
    
        img_y = cv2.imread(image_pathY)
        img_u = cv2.imread(image_pathU)
        img_v = cv2.imread(image_pathV)
    
        result_y = shuffleImage(img_y, key1)
        cv2.imwrite(image_pathY, result_y)
        result_u = shuffleImage(img_u, key1)
        cv2.imwrite(image_pathU, result_u)
        result_v = shuffleImage(img_v, key1)
        cv2.imwrite(image_pathV, result_v)
    
    
    # Parity Matrix H (required for error correcting)
    H = [
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [0, 1, 1],
         [1, 1, 1],
         [1, 0, 1]
    ]
    message_retrieved = np.zeros(len_y, dtype = np.uint8)
    message_len = 0  ### so we move the pixels of the 1D array from above
    
    # To get the dimension of one frame
    frame_path = os.path.join(workingDir, 'framesY', '0.png')
    Y = cv2.imread(frame_path)
    height, width, channels = Y.shape  ### we need width
    dimension = height * width - 1
    
    # To see how many Pixels we need to read in each frame
    # Last frame can have a few more pixels
    pixels_in_frame = (len_y // 4) // frame_count
    pixels_for_last_frame = (len_y // 4) % frame_count
    
    # (Key1 ^ Key2) % 2 == 0 -> beginning, then end; otherwise, reverse 
    parity = 0 # 1 -> begin; -1 -> end 
    if (key1 ^ key2) % 2 == 0:
        parity = 1
    else:
        parity = -1
    
    # Aux will contain the Hamming code extracted from frame
    aux = [0, 0, 0, 0, 0, 0, 0] 
    # From each frame, extract the modified Pixels
    for i in range(frame_count):
        # If last frame, we might have more Pixels
        if i == frame_count - 1:
            pixels_in_frame += pixels_for_last_frame
        
        # Open the frames from which we will read the pixels
        name = str(i) + '.png'
        path_y = os.path.join(workingDir, 'framesY', name)
        Y = cv2.imread(path_y)
        path_u = os.path.join(workingDir, 'framesU', name)
        U = cv2.imread(path_u)
        path_v = os.path.join(workingDir, 'framesV', name)
        V = cv2.imread(path_v)
        
        ### We read the modified Pixels
        for j in range(pixels_in_frame):
            if parity == 1:
                aux[0] = Y[j // width][j % width][0] % 2
                aux[1] = Y[j // width][j % width][1] % 2
                aux[2] = Y[j // width][j % width][2] % 2
    
                aux[3] = U[j // width][j % width][0] % 2
                aux[4] = U[j // width][j % width][1] % 2
    
                aux[5] = V[j // width][j % width][0] % 2
                aux[6] = V[j // width][j % width][1] % 2
    
                parity = -1
            else:
                contor = dimension - j
                aux[0] = Y[contor // width][contor % width][0] % 2
                aux[1] = Y[contor // width][contor % width][1] % 2
                aux[2] = Y[contor // width][contor % width][2] % 2
    
                aux[3] = U[contor // width][contor % width][0] % 2
                aux[4] = U[contor // width][contor % width][1] % 2
    
                aux[5] = V[contor // width][contor % width][0] % 2
                aux[6] = V[contor // width][contor % width][1] % 2
    
                parity = 1
    
    
            # Error correcting
            Z = np.matmul(aux, H)
            Z[0] %= 2
            Z[1] %= 2
            Z[2] %= 2
    
            # No error (all 0-s)
            if Z[0] == 0 and Z[1] == 0 and Z[2] == 0:
                message_retrieved[message_len] = aux[3] * 255
                message_len += 1
                message_retrieved[message_len] = aux[4] * 255
                message_len += 1
                message_retrieved[message_len] = aux[5] * 255
                message_len += 1
                message_retrieved[message_len] = aux[6] * 255
                message_len += 1
            elif Z[0] == 1 and Z[1] == 1 and Z[2] == 0:
                message_retrieved[message_len] = ((aux[3] + 1) % 2) * 255
                message_len += 1
                message_retrieved[message_len] = aux[4] * 255
                message_len += 1
                message_retrieved[message_len] = aux[5] * 255
                message_len += 1
                message_retrieved[message_len] = aux[6] * 255
                message_len += 1
            elif Z[0] == 0 and Z[1] == 1 and Z[2] == 1:
                message_retrieved[message_len] = aux[3] * 255
                message_len += 1
                message_retrieved[message_len] = ((aux[4] + 1) % 2) * 255
                message_len += 1
                message_retrieved[message_len] = aux[5] * 255
                message_len += 1
                message_retrieved[message_len] = aux[6] * 255
                message_len += 1
            elif Z[0] == 1 and Z[1] == 1 and Z[2] == 1:
                message_retrieved[message_len] = aux[3] * 255
                message_len += 1
                message_retrieved[message_len] = aux[4] * 255
                message_len += 1
                message_retrieved[message_len] = ((aux[5] + 1) % 2) * 255
                message_len += 1
                message_retrieved[message_len] = aux[6] * 255
                message_len += 1
            elif Z[0] == 1 and Z[1] == 0 and Z[2] == 1:
                message_retrieved[message_len] = aux[3] * 255
                message_len += 1
                message_retrieved[message_len] = aux[4] * 255
                message_len += 1
                message_retrieved[message_len] = aux[5] * 255
                message_len += 1
                message_retrieved[message_len] = ((aux[6] + 1) % 2) * 255
                message_len += 1
        
        # Close the previous YUV frames
        name = str(i) + '.png'
        path_y = os.path.join(workingDir, 'framesY', name)
        cv2.imwrite(path_y, Y)
        path_u = os.path.join(workingDir, 'framesU', name)
        cv2.imwrite(path_u, U)
        path_v = os.path.join(workingDir, 'framesV', name)
        cv2.imwrite(path_v, V)
    
    name = str(i - 1) + '.png'
    path_y = os.path.join(workingDir, 'framesY', name)
    cv2.imwrite(path_y, Y)
    path_u = os.path.join(workingDir, 'framesU', name)
    cv2.imwrite(path_u, U)
    path_v = os.path.join(workingDir, 'framesV', name)
    cv2.imwrite(path_v, V)
    
    # Retrieve the hidden photo
    img = message_retrieved.reshape((height_photo, width_photo))
    cv2.imwrite(output_path, img)
    
    img_y = cv2.imread(output_path)
    result_y = unshuffleImage(img_y, key2)
    cv2.imwrite(output_path, result_y)


 
###############################################################################   
### Decode Main Function (Video Stegano)
def decode(video_file_path, key3, len_data, output_path):
    key1, key2, width, height, len_y = process_stego_audio(video_file_path, key3, len_data, output_path)
    if key1 == -1:
        return
    else:
        extract_hidden_photo(video_file_path, key1, key2, width, height, len_y, output_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    