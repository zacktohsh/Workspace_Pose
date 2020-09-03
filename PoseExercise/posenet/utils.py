import cv2
import numpy as np

import posenet.constants

import math


NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_HAND = 9
RIGHT_HAND = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

PUT_TEXT_SIZE = 2
THICKNESS = 7

FONT_COLOR = (255,0,0)
FONT_COLOR_R = (0,0,255)

OFFSET = 280


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    elbow_high = 0
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    body_coords = ["","","","","","","","","","","","","","","","",""]
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)
        #print(new_keypoints)

        #1st point is nose
        #2nd point is left eye
        #3rd point is right eye
        #4th point is left ear
        #5th point is right ear
        #6th point is left shoulder
        #7th point is right shoulder
        count = 0
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                count +=1
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

            if count == LEFT_EYE:
                body_coords[LEFT_EYE] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_EYE:
                body_coords[RIGHT_EYE] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_EAR:
                body_coords[LEFT_EAR] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_EAR:
                body_coords[RIGHT_EAR] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_SHOULDER:
                body_coords[LEFT_SHOULDER] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_SHOULDER:
                body_coords[RIGHT_SHOULDER] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_ELBOW:
                body_coords[LEFT_ELBOW] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_ELBOW:
                body_coords[RIGHT_ELBOW] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_HAND:
                body_coords[LEFT_HAND] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_HAND:
                body_coords[RIGHT_HAND] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_HIP:
                body_coords[LEFT_HIP] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_HIP:
                body_coords[RIGHT_HIP] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_KNEE:
                body_coords[LEFT_KNEE] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_KNEE:
                body_coords[RIGHT_KNEE] = (int(kc[1])-OFFSET,int(kc[0]))
            if count == LEFT_ANKLE:
                body_coords[LEFT_ANKLE] = (int(kc[1]),int(kc[0]))
            if count == RIGHT_ANKLE:
                body_coords[RIGHT_ANKLE] = (int(kc[1])-OFFSET,int(kc[0]))
            count +=1
			
    if cv_keypoints:
        #print(cv_keypoints)
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    #'''
    if body_coords[LEFT_EYE]:
        out_img = cv2.putText(out_img,"Left Eye",body_coords[LEFT_EYE], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_EYE]:
        out_img = cv2.putText(out_img,"Right Eye",body_coords[RIGHT_EYE], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_EAR]:
        out_img = cv2.putText(out_img,"Left Ear",body_coords[LEFT_EAR], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_EAR]:
        out_img = cv2.putText(out_img,"Right Ear",body_coords[RIGHT_EAR], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_SHOULDER]:
        out_img = cv2.putText(out_img,"Right Sho",body_coords[LEFT_SHOULDER], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_SHOULDER]:
        out_img = cv2.putText(out_img,"Right Sho",body_coords[RIGHT_SHOULDER], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_ELBOW]:
        out_img = cv2.putText(out_img,"Left Elb",body_coords[LEFT_ELBOW], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_ELBOW]:
        out_img = cv2.putText(out_img,"Right Elb",body_coords[RIGHT_ELBOW], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_HAND]:
        out_img = cv2.putText(out_img,"Left Han",body_coords[LEFT_HAND], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_HAND]:
        out_img = cv2.putText(out_img,"Right Han",body_coords[RIGHT_HAND], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_HIP]:
        out_img = cv2.putText(out_img,"Left Hip",body_coords[LEFT_HIP], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_HIP]:
        out_img = cv2.putText(out_img,"Right Hip",body_coords[RIGHT_HIP], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_KNEE]:
        out_img = cv2.putText(out_img,"Left Kne",body_coords[LEFT_KNEE], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_KNEE]:
        out_img = cv2.putText(out_img,"Right Kne",body_coords[RIGHT_KNEE], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    if body_coords[LEFT_ANKLE]:
        out_img = cv2.putText(out_img,"Left Ank",body_coords[LEFT_ANKLE], font, PUT_TEXT_SIZE,FONT_COLOR,THICKNESS,cv2.LINE_AA)
    if body_coords[RIGHT_ANKLE]:
        out_img = cv2.putText(out_img,"Right Ank",body_coords[RIGHT_ANKLE], font, PUT_TEXT_SIZE,FONT_COLOR_R,THICKNESS,cv2.LINE_AA)
    #'''
    #return out_img
    '''
    if body_coords[LEFT_ELBOW] and body_coords[LEFT_SHOULDER]:
      if body_coords[LEFT_ELBOW][1] < body_coords[LEFT_SHOULDER][1]:
          print("Left Elbow Higher Than Shoulder!")
      if body_coords[LEFT_ELBOW][1] > body_coords[LEFT_SHOULDER][1]:
          print("Left Elbow Lower Than Shoulder!")
    '''
    return out_img
