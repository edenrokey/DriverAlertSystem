# Import Libraries
import time
import math
from collections import deque
import cProfile

import sys
import numpy
import cv2

import Image
import ImageOps
import ImageEnhance


# Constants
CAMERA_INDEX = 0;
SCALE_FACTOR = 5; # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml";
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml";
FACE_MIN_SIZE = 0.2;
EYE_MIN_SIZE = 0.03;

DISPLAY_SCALE = 0.3333;
FACE_SCALE = 0.25;
EYE_SCALE = 0.33333;

class FaceDetector:

	"""
	FaceDetector is a wrapper for the cascade classifiers.
	Must be initialized using faceClassifierPath and eyeClassifierPath, and 
	should only be initialized once per program instance. The only "public"
	method is detect().
	"""

	def __init__(self, faceClassifierPath, eyeClassifierPath):
		"""
		Initialize & Load Haar Cascade Classifiers.
		
		Args:
			faceClassifierPath (string): path to face Haar classifier
			eyeClassifierPath (string): path to eye Haar classifier
		"""
		self.faceClassifier = cv2.CascadeClassifier(faceClassifierPath);
		self.eyeClassifier = cv2.CascadeClassifier(eyeClassifierPath);
	
	def detect(self,frames, faceRect=False):
		"""
		Detect face and eyes. 
		Runs Haar cascade classifiers. Sometimes it is desirable to speed up 
		processing by using a previously-found face rectangle. To do this, pass 
		the old faceRect as the second argument.
		
		Args:
			frames (dict of numpy array): dictionary containing images with different scales
			faceRect (numpy array): array of face rectangle. Face detected if 
									 omitted.
		Returns:
			a dictionary with three elements each representing a rectangle
		"""
# Data structure to hold frame info
		rects = {
			'face': numpy.array([],dtype=numpy.int32),
			'eyeLeft': numpy.array([],dtype=numpy.int32),
			'eyeRight': numpy.array([],dtype=numpy.int32)
		};
		
# Detect face if old faceRect not provided
		if faceRect is False or len(faceRect) is 0:
			faceIMG = frames['face'];
			faceRects = self.classifyFace(faceIMG);
			
			# Ensure a single face found
			if len(faceRects) is 1:
				faceRect = faceRects[0];
			else:
				# TODO throw error message
				print "No Faces / Multiple Faces Found!";
				return rects;
			
		rects['face'] = faceRect;

# Extract face coordinates, calculate center and diameter
		x1,y1,x2,y2 = rects['face'];
		faceCenter = (((x1+x2)/2.0), ((y1+y2)/2.0));
		faceDiameter = y2-y1;
		
# Extract eyes region of interest (ROI), cropping mouth and hair
		eyeBBox = numpy.array([x1,
		                      (y1 + (faceDiameter*0.24)),
		                      x2,
		                      (y2 - (faceDiameter*0.40))],dtype=numpy.int32);

# Search for eyes in ROI
		eyeRects = self.classifyEyes(frames['eyes'],eyeBBox);
		print eyeRects;
		if len(eyeRects) < 2:
			print "blinking_of_eye"

# Loop over each eye
		for e in eyeRects:
# Adjust coordinates to be in faceRect's coordinate space
#			e += numpy.array([eyesX1, eyesY1, eyesX1, eyesY1],dtype=numpy.int32);
						
# Split left and right eyes. Compare eye and face midpoints.
			eyeMidpointX = (e[0]+e[2])/2.0;
			if eyeMidpointX < faceCenter[0]:
				rects['eyeLeft'] = e; # TODO prevent overwriting
			else:
				rects['eyeRight'] = e;
		
		return rects;

	def classify(self, img, cascade, minSizeX=40):
		"""Run Cascade Classifier on Image"""
		minSizeX = int(round(minSizeX));
#		print 'minSizeX:',minSizeX
		# Run Cascade Classifier
		rects = cascade.detectMultiScale(
				img, minSize=(minSizeX,minSizeX), 
				flags=cv2.CASCADE_SCALE_IMAGE);
		
# No Results
		if len(rects) == 0:
			return numpy.array([],dtype=numpy.int32);
		
		rects[:,2:] += rects[:,:2]; 
		rects = numpy.array(rects,dtype=numpy.int32);
		return rects;
	
	def classifyFace(self,img):
		"""Run Face Cascade Classifier on Image"""
		rects = self.classify(img,self.faceClassifier,img.shape[1]*FACE_MIN_SIZE);
		return rects/FACE_SCALE;
	
	def classifyEyes(self,img,bBox):
		"""Run Eyes Cascade Classifier on Image"""
		EYE_MIN_SIZE = 0.15;
		bBoxScaled = bBox*EYE_SCALE;
		eyesROI = img[bBoxScaled[1]:bBoxScaled[3], bBoxScaled[0]:bBoxScaled[2]];
		
		eyesROI = cv2.equalizeHist(eyesROI);
		
#		print 'eyesROI dimensions: ',eyesROI.shape;
		minEyeSize = eyesROI.shape[1]*EYE_MIN_SIZE;
#		print 'minEyeSize:',minEyeSize;
		cv2.imshow("eyesROI",eyesROI);
		cv2.moveWindow("eyeROI",150,300);
		rectsScaled = self.classify(eyesROI, self.eyeClassifier, 
									minEyeSize);
		
#		print rectsScaled;
# Scale back to full size
		rects = rectsScaled / EYE_SCALE;
		
# Loop over each eye
		for eye in rects:
			# Adjust coordinates to be in faceRect's coordinate space
			eye += numpy.array([bBox[0],bBox[1],bBox[0],bBox[1]]);

		return rects;



class FaceModel:

	"""
	FaceModel integrates data from the new frame into a model that keeps track of where the eyes are. To do this it uses:
		- A moving average of the most recent frames
		- Facial geometry to fill in missing data
	The resulting model generates a set of two specific regions of interest (ROI's) where blinking is expected to take place.
	"""

	QUEUE_MAXLEN = 50;
	
	QUALITY_QUEUE_MAXLEN = 30;
	qualityHistory = {
		'face':deque(maxlen=QUALITY_QUEUE_MAXLEN),
		'eyeLeft':deque(maxlen=QUALITY_QUEUE_MAXLEN),
		'eyeRight':deque(maxlen=QUALITY_QUEUE_MAXLEN)
	};
	
# Queues storing most recent position rectangles, used to calculate moving averages
	rectHistory = {
		'face': deque(maxlen=QUEUE_MAXLEN),
		'eyeLeft': deque(maxlen=QUEUE_MAXLEN),
		'eyeRight': deque(maxlen=QUEUE_MAXLEN)
	};
	
#Moving average of position rectangles
	rectAverage = {
		'face': numpy.array([]),
		'eyeLeft': numpy.array([]),
		'eyeRight': numpy.array([])
	};
	
	def add(self,rects):
		"""Add new set of rectangles to model"""
		
		# Checks to see if face has moved significantly. If so, resets history.
		if(self._faceHasMoved(rects['face'])):
			self.clear();
				
		# Loop over rectangles, adding non-empty ones to history
		for key,rect in rects.items():
			if len(rect) is not 4:
				self.qualityHistory[key].append(0);
				continue;
			self.rectHistory[key].append(rect);
			self.qualityHistory[key].append(1);
#			print 'appended to qHist[',key,']';
		
# Update moving average stats
		self._updateAverages();

	def getPreviousFaceRects(self):
		if len(self.rectHistory['face']) is 0:
			return numpy.array([],dtype=numpy.int32);
		else:
			return self.rectHistory['face'][-1];
	
	def getEyeRects(self):
		"""Get array of eye rectangles"""
		return [self.rectAverage['eyeLeft'], self.rectAverage['eyeRight']];
	
	def getFaceRect(self):
		"""Get face rectangle"""
		return self.rectAverage['face'];

	def getEyeLine(self):
		"""Returns Points to create line along axis of eyes"""
		left,right = self.getEyeRects();
		
		if len(left) is not 4 or len(right) is not 4:
			return [(0,0),(0,0)];
		
		leftPoint = (left[0], ((left[1] + left[3])/2));
		rightPoint = (right[2], ((right[1] + right[3])/2));
		return [leftPoint,rightPoint];
		
	def clear(self):
		""" Resets Eye History"""
		for key,value in self.rectAverage.items():
			self.rectAverage[key] = numpy.array([],dtype=numpy.int32);
			self.rectHistory[key].clear();
			self.qualityHistory[key].clear();

	def _faceHasMoved(self, recentFaceRect):
		"""Determines if face has just moved, requiring history reset"""
	
		# If no face found, return true
		if(len(recentFaceRect) is not 4):
			return True;

		history = self.rectHistory['face'];
		
		if len(history) is not self.QUEUE_MAXLEN:
			return False;

		old = history[self.QUEUE_MAXLEN - 10];
		oldX = (old[0] + old[2]) / 2.0;
		oldY = (old[1] + old[3]) / 2.0;
		recentX = (recentFaceRect[0] + recentFaceRect[2]) / 2.0;
		recentY = (recentFaceRect[1] + recentFaceRect[3]) / 2.0;
		change = ((recentX-oldX)**2 + (recentY-oldY)**2)**0.5; # sqrt(a^2+b^2)
		return True if change > 15 else False;

	def _updateAverages(self):
		"""Update position rectangle moving averages"""
		for key,queue in self.rectHistory.items():
			if len(queue) is 0:
				continue;
			self.rectAverage[key] = sum(queue) / len(queue);
		
		faceQ = numpy.mean(self.qualityHistory['face']);
		eyeLeftQ = numpy.mean(self.qualityHistory['eyeLeft']);
		eyeRightQ = numpy.mean(self.qualityHistory['eyeRight']);


class Util:

	@staticmethod
	def contrast(img, amount='auto'):
		"""
		Modify image contrast
		
		Args:
			img (numpy array)			Input image array
			amount (float or string)  	Either number (e.g. 1.3) or 'auto'
		"""
		
		pilIMG = Image.fromarray(img);
		
		if amount is 'auto':
			pilEnhancedIMG = ImageOps.autocontrast(pilIMG, cutoff = 0);
			return numpy.asarray(pilEnhancedIMG);
		else:
			pilContrast = ImageEnhance.Contrast(pilIMG);
			pilContrasted = pilContrast.enhance(amount);
			return numpy.asarray(pilContrasted);

	@staticmethod
	def threshold(img, thresh):
		"""Threshold an image"""
		
		pilIMG1 = Image.fromarray(img);
		pilInverted1 = ImageOps.invert(pilIMG1);
		inverted = numpy.asarray(pilInverted1);
		r, t = cv2.threshold(inverted, thresh, 0, type=cv2.THRESH_TOZERO);
		pilIMG2 = Image.fromarray(t);
		pilInverted2 = ImageOps.invert(pilIMG2);
		thresholded = numpy.asarray(pilInverted2);
		return thresholded;

	
	@staticmethod
	def equalizeHSV(img, equalizeH=False, equalizeS=False, equalizeV=True):
		"""
		Equalize histogram of color image using BSG2HSV conversion
		By default only equalizes the value channel
		
		Note: OpenCV's HSV implementation doesn't capture all hue info, see:
		http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CvtColor
		http://www.shervinemami.info/colorConversion.html
		"""

		imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);
		h,s,v = cv2.split(imgHSV);
		
		if equalizeH:
			h = cv2.equalizeHist(h);
		if equalizeS:
			s = cv2.equalizeHist(s);
		if equalizeV:
			v = cv2.equalizeHist(v);
		
		hsv = cv2.merge([h,s,v]);
		bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
		return bgr;


class Display:

	def renderScene(self, frame, model, rects=False):
		"""Draw face and eyes onto image, then display it"""
		
		# Get Coordinates
		eyeRects = model.getEyeRects();
		faceRect = model.getFaceRect();
		linePoints = model.getEyeLine();
	
		# Draw Shapes and display frame
		self.drawLine(frame, linePoints[0],linePoints[1],(0, 0, 255));
		self.drawRectangle(frame, faceRect, (0, 0, 255));
		self.drawRectangle(frame, eyeRects[0], (0, 255, 0));
		self.drawRectangle(frame, eyeRects[1], (0, 255, 0));
		
		if rects is not False:
			self.drawRectangle(frame, rects['eyeLeft'], (152,251,152));
			self.drawRectangle(frame, rects['eyeRight'],(152,251,152));
		
		cv2.imshow("Video", frame);
	
	@staticmethod
	def drawLine(img, p1, p2, color):
		"""Draw lines on image"""
		p1 = (int(p1[0]*DISPLAY_SCALE), int(p1[1]*DISPLAY_SCALE));
		p2 = (int(p2[0]*DISPLAY_SCALE), int(p2[1]*DISPLAY_SCALE));
		cv2.line(img, p1, p2,(0, 0, 255));
	
	@staticmethod
	def drawRectangle(img, rect, color):
		"""Draw rectangles on image"""
		
		if len(rect) is not 4:
			
			return;
		rect = rect * DISPLAY_SCALE;
		x1, y1, x2, y2 = rect.astype(numpy.int32);
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2);


class Capture:

	camera = cv2.VideoCapture(0);
	height = 0;
	width = 0;
	
	def __init__(self, scaleFactor=1):
	
		# Setup webcam dimensions
		self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT);
		self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH);
		
		# Reduce Video Size to make Processing Faster
		if scaleFactor is not 1:
			scaledHeight = self.height / scaleFactor;
			scaledWidth = self.width / scaleFactor;
			self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT,scaledHeight);
			self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,scaledWidth);
	
		# Create window
		cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE);
	
	def read(self):
		retVal, colorFrame = self.camera.read();
		displayFrame = cv2.resize(colorFrame,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE);
		
		grayFrame = cv2.equalizeHist(cv2.cvtColor(colorFrame,cv2.COLOR_BGR2GRAY));
		
		faceFrame = cv2.resize(grayFrame,None,fx=FACE_SCALE,fy=FACE_SCALE);
		
		eyesFrame = cv2.resize(cv2.equalizeHist(cv2.cvtColor(colorFrame,cv2.COLOR_BGR2GRAY)),None,fx=EYE_SCALE,fy=EYE_SCALE);
		
		frames = {
			'color': colorFrame,
			'display': displayFrame,
			#'gray': grayFrame,
			'face': faceFrame,
			'eyes': eyesFrame
		};
		
		return frames;

def main():
	# Instantiate Classes
	detector = FaceDetector(FACE_CLASSIFIER_PATH, EYE_CLASSIFIER_PATH);
	model = FaceModel();
	display = Display();
	capture = Capture();
	
	oldTime = time.time();
	i = 0;
	
	while True:
		# Calculate time difference (dt), update oldTime variable
		newTime = time.time();
		dt =  newTime - oldTime;
		oldTime = newTime;
		
		# Grab Frames
		frames = capture.read();	
		
		# Detect face 20% of the time, eyes 100% of the time
		if i % 5 is 0:
			rects = detector.detect(frames);
		else:
			rects = detector.detect(frames,model.getPreviousFaceRects());
		i += 1;
	
		# Add detected rectangles to model
		model.add(rects);
		
		# Render
		cv2.imshow("Video", frames['display']);#displayFrame);
		display.renderScene(frames['display'],model,rects);
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
   		   break
# When everything is done, release the capture
cv2.destroyAllWindows()

cProfile.run('main()','profile.o','cumtime');






		




