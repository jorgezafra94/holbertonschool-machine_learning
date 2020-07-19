# Face Verification
* Here we are going to use a pre trained model a face_verification.h5 model
* the images of our classmates of cohort8 in Holberton
* we are going to use a csv file to get the triplets
* And a predictor that holberton gave us
## Task0 - Load Images
Write the function `def load_images(images_path, as_array=True):` that loads images from a directory or file:<br>
<br>
* images_path is the path to a directory from which to load images
* as_array is a boolean indicating whether the images should be loaded as one numpy.ndarray
* If True, the images should be loaded as a numpy.ndarray of shape (m, h, w, c) where:
* * m is the number of images
* * h, w, and c are the height, width, and number of channels of all images, respectively
* If False, the images should be loaded as a list of individual numpy.ndarrays
* All images should be loaded in RGB format
* The images should be loaded in alphabetical order by filename<br>

Returns: images, filenames
* images is either a list/numpy.ndarray of all images
* filenames is a list of the filenames associated with each image in images<br>

```
ubuntu-xenial:0x0B-face_verification$ cat 0-main.py
#!/usr/bin/env python3

from utils import load_images
import matplotlib.pyplot as plt

images, filenames = load_images('HBTN', as_array=False)
print(type(images), len(images))
print(type(filenames), len(filenames))
idx = filenames.index('KirenSrinivasan.jpg')
print(idx)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i+idx])
    plt.title(filenames[i+idx])
plt.tight_layout()
plt.show()

ubuntu-xenial:0x0B-face_verification$ ./0-main.py
<class 'list'> 385
<class 'list'> 385
195
```

![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/loading_imgs.png)<br>
## Task1 - Load CSV
Also in utils.py, write a function `def load_csv(csv_path, params={}):` that loads the contents of a csv file as a list of lists:<br>
<br>
* csv_path is the path to the csv to load
* params are the parameters to load the csv with<br>

Returns: a list of lists representing the contents found in csv_path

```
ubuntu-xenial:0x0B-face_verification$ cat 1-main.py
#!/usr/bin/env python3

from utils import load_csv

triplets = load_csv('FVTriplets.csv')
print(type(triplets), len(triplets))
print(triplets[:10])
ubuntu-xenial:0x0B-face_verification$ ./1-main.py
<class 'list'> 5306
[['AndrewMaring', 'AndrewMaring0', 'ArthurDamm0'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm1'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm2'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm3'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm4'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm5'], ['AndrewMaring', 'AndrewMaring0', 'ArthurDamm6'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm0'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm1'], ['AndrewMaring', 'AndrewMaring1', 'ArthurDamm2']]
ubuntu-xenial:0x0B-face_verification$
```

## Task2 - Initialize Face Align 
Create the class FaceAlign:<br>
<br>
class constructor `def __init__(self, shape_predictor_path):`
* shape_predictor_path is the path to the dlib shape predictor model
* Sets the public instance attributes:
* detector - contains dlib‘s default face detector
* shape_predictor - contains the dlib.shape_predictor

```
ubuntu-xenial:0x0B-face_verification$ cat 2-main.py
#!/usr/bin/env python3

from align import FaceAlign

fa = FaceAlign('models/landmarks.dat')
print(type(fa.detector))
print(type(fa.shape_predictor))

ubuntu-xenial:0x0B-face_verification$ ./2-main.py
<class 'dlib.fhog_object_detector'>
<class 'dlib.shape_predictor'>
ubuntu-xenial:0x0B-face_verification$
```

## Task3 - Detect Faces
Update the class FaceAlign:<br>
<br>
public instance method `def detect(self, image):` that detects a face in an image:
* image is a numpy.ndarray of rank 3 containing an image from which to detect a face<br>

Returns: a dlib.rectangle containing the boundary box for the face in the image, or None on failure
* If multiple faces are detected, return the dlib.rectangle with the largest area
* If no faces are detected, return a dlib.rectangle that is the same as the image

```
ubuntu-xenial:0x0B-face_verification$ cat 3-main.py
#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
box = fa.detect(test_img)
print(type(box))
plt.imshow(test_img)
ax = plt.gca()
rect = Rectangle((box.left(), box.top()), box.width(), box.height(), fill=False)
ax.add_patch(rect)
plt.show()

ubuntu-xenial:0x0B-face_verification$ ./3-main.py
<class 'dlib.rectangle'>
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/detecting_face.png)

## Task4 - Find Landmarks
Update the class FaceAlign:<br>
<br>
public instance method `def find_landmarks(self, image, detection):` that finds facial landmarks:
* image is a numpy.ndarray of an image from which to find facial landmarks
* detection is a dlib.rectangle containing the boundary box of the face in the image<br>

Returns: a numpy.ndarray of shape (p, 2)containing the landmark points, or None on failure
* p is the number of landmark points
* 2 is the x and y coordinates of the point

```
ubuntu-xenial:0x0B-face_verification$ cat 4-main.py
#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
box = fa.detect(test_img)
landmarks = fa.find_landmarks(test_img, box)
print(type(landmarks), landmarks.shape)
plt.imshow(test_img)
ax = plt.gca()
for landmark in landmarks:
    ax.add_patch(Circle(landmark))
plt.show()

ubuntu-xenial:0x0B-face_verification$ ./4-main.py
<class 'numpy.ndarray'> (68, 2)
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/landmarks_face.png)

## Task5 - Align Faces
Update the class FaceAlign:<br>
<br>
public instance method `def align(self, image, landmark_indices, anchor_points, size=96):` that aligns an image for face verification:
* image is a numpy.ndarray of rank 3 containing the image to be aligned
* landmark_indices is a numpy.ndarray of shape (3,) containing the indices of the three landmark points that should be used for the affine transformation
* anchor_points is a numpy.ndarray of shape (3, 2) containing the destination points for the affine transformation, scaled to the range [0, 1]
* size is the desired size of the aligned image<br>

Returns: a numpy.ndarray of shape (size, size, 3) containing the aligned image, or None if no face is detected

```
ubuntu-xenial:0x0B-face_verification$ cat 5-main.py
#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
aligned = fa.align(test_img, np.array([36, 45, 33]), anchors, 96)
plt.imshow(aligned)
ax = plt.gca()
for anchor in anchors:
    ax.add_patch(Circle(anchor * 96, 1))
plt.show()

ubuntu-xenial:0x0B-face_verification$ ./5-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/processed_image.png)

## Task6 - Save Images
Also in utils.py, write a function `def save_images(path, images, filenames):` that saves images to a specific path:<br>
<br>
* path is the path to the directory in which the images should be saved
* images is a list/numpy.ndarray of images to save
* filenames is a list of filenames of the images to save<br>

Returns: True on success and False on failure

```
ubuntu-xenial:0x0B-face_verification$ cat 6-main.py
#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_images, save_images

fa = FaceAlign('models/landmarks.dat')
images, filenames = load_images('HBTN', as_array=False)
anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
aligned = []
for image in images:
    aligned.append(fa.align(image, np.array([36, 45, 33]), anchors, 96))
aligned = np.array(aligned)
print(aligned.shape)
if not os.path.isdir('HBTNaligned'):
    print(save_images('HBTNaligned', aligned, filenames))
    os.mkdir('HBTNaligned')
print(save_images('HBTNaligned', aligned, filenames))
print(os.listdir('HBTNaligned'))
image = plt.imread('HBTNaligned/KirenSrinivasan.jpg')
plt.imshow(image)
plt.show()

ubuntu-xenial:0x0B-face_verification$ ./6-main.py
False
True
['MariaCoyUlloa4.jpg', 'TuVo0.jpg', 'XimenaCarolinaAndradeVargas1.jpg', 'RodrigoCruz4.jpg', 'LeineValente0.jpg', 'JuanValencia1.jpg', 'DennisPham3.jpg', 'NgaLa3.jpg', 'RodrigoCruz0.jpg', 'LeineValente4.jpg', 'HeimerRojas5.jpg', 'LauraRoudge5.jpg', 'FaizanKhan2.jpg', 'KennethCortesAguas4.jpg', 'AdamSedki3.jpg', 'FaizanKhan1.jpg', 'KennethCortesAguas1.jpg', 'FrancescaCantor6.jpg', 'SamieAzad3.jpg', 'DavidKwan2.jpg', 'DiegoAndrésCastellanosRodríguez0.jpg', 'JulienneTesoro6.jpg', 'PhuTruong2.jpg', 'JohnCook4.jpg', 'RussellMolimock1.jpg', 'SnehaDasaLakshminath3.jpg', 'AnthonyLe1.jpg', 'AndrewMaring3.jpg', 'YesidGonzalez1.jpg', 'HeimerRojas6.jpg', 'RussellMolimock4.jpg', 'DiegoAndrésCastellanosRodríguez1.jpg', 'AdamSedki.jpg', 'NgaLa2.jpg', 'LauraVasquezBernal.jpg', 'CarlosArias5.jpg', 'FrancescaCantor2.jpg', 'ArthurDamm1.jpg', 'DennisPham6.jpg', 'BrentJanski0.jpg', 'XimenaCarolinaAndradeVargas.jpg', 'OlgaLucíaRodríguezToloza0.jpg', 'FaizanKhan0.jpg', 'LeoByeon0.jpg', 'ElaineYeung4.jpg', 'ChristianWilliams0.jpg', 'JohnCook3.jpg', 'KoomeMwiti0.jpg', 'DavidKwan.jpg', 'JuanValencia4.jpg', 'RodrigoCruz1.jpg', 'DavidKwan3.jpg', 'JaiberRamirez3.jpg', 'AnthonyLe2.jpg', 'RodrigoCruz.jpg', 'TimAssavarat2.jpg', 'SamuelAlexanderFlorez7.jpg', 'SamieAzad0.jpg', 'FrancescaCantor4.jpg', 'CarlosArias4.jpg', 'JavierCañon5.jpg', 'GiovannyAlexanderRubioAlbornoz1.jpg', 'HeimerRojas3.jpg', 'RussellMolimock0.jpg', 'NgaLa5.jpg', 'KyleLitscher3.jpg', 'PhuTruong3.jpg', 'AlishaSmith3.jpg', 'JulienneTesoro5.jpg', 'TuVo2.jpg', 'SamieAzad2.jpg', 'AlishaSmith0.jpg', 'SamuelAlexanderFlorez.jpg', 'BrentJanski2.jpg', 'KennethCortesAguas.jpg', 'DennisPham7.jpg', 'XimenaCarolinaAndradeVargas6.jpg', 'ArthurDamm.jpg', 'RyanHudson.jpg', 'YesidGonzalez4.jpg', 'LauraVasquezBernal0.jpg', 'SamieAzad5.jpg', 'DianaBoada6.jpg', 'NgaLa.jpg', 'DiegoAndrésCastellanosRodríguez4.jpg', 'HaroldoVélezLora4.jpg', 'KoomeMwiti4.jpg', 'MiaMorton1.jpg', 'AlishaSmith2.jpg', 'NgaLa4.jpg', 'XimenaCarolinaAndradeVargas0.jpg', 'DianaBoada1.jpg', 'TuVo4.jpg', 'MarkHedgeland1.jpg', 'SofiaCheung2.jpg', 'RicardoGutierrez6.jpg', 'BrentJanski1.jpg', 'CarlosArias0.jpg', 'RodrigoCruz5.jpg', 'TuVo3.jpg', 'LauraVasquezBernal4.jpg', 'XimenaCarolinaAndradeVargas4.jpg', 'BrentJanski6.jpg', 'ElaineYeung3.jpg', 'DianaBoada3.jpg', 'SamieAzad6.jpg', 'SamuelAlexanderFlorez3.jpg', 'RodrigoCruz7.jpg', 'RicardoGutierrez4.jpg', 'HeimerRojas2.jpg', 'KoomeMwiti.jpg', 'FaizanKhan3.jpg', 'JuanValencia3.jpg', 'TimAssavarat.jpg', 'JaiberRamirez1.jpg', 'SnehaDasaLakshminath2.jpg', 'LeineValente2.jpg', 'SamieAzad1.jpg', 'HaroldoVélezLora0.jpg', 'ElaineYeung7.jpg', 'HaroldoVélezLora.jpg', 'CarlosArias.jpg', 'GiovannyAlexanderRubioAlbornoz5.jpg', 'BrentJanski4.jpg', 'HongtuHuang.jpg', 'RodrigoCruz2.jpg', 'HeimerRojas.jpg', 'AlishaSmith.jpg', 'NgaLa0.jpg', 'PhuTruong0.jpg', 'JulienneTesoro2.jpg', 'ArthurDamm2.jpg', 'BrendanEliason4.jpg', 'XimenaCarolinaAndradeVargas5.jpg', 'JaiberRamirez7.jpg', 'AndrewMaring1.jpg', 'MohamethSeck0.jpg', 'AllisonWeiner.jpg', 'KirenSrinivasan0.jpg', 'LeineValente1.jpg', 'DennisPham.jpg', 'SamuelAlexanderFlorez2.jpg', 'AndrewMaring.jpg', 'BrendanEliason3.jpg', 'HeimerRojas1.jpg', 'RicardoGutierrez8.jpg', 'MohamethSeck2.jpg', 'KoomeMwiti3.jpg', 'MohamethSeck6.jpg', 'CarlosArias2.jpg', 'ElaineYeung.jpg', 'JohnCook1.jpg', 'JaiberRamirez4.jpg', 'ElaineYeung6.jpg', 'LeoByeon.jpg', 'BrendanEliason1.jpg', 'RussellMolimock3.jpg', 'KirenSrinivasan.jpg', 'RobertSebastianCastellanosRodriguez.jpg', 'RicardoGutierrez5.jpg', 'RicardoGutierrez3.jpg', 'RicardoGutierrez1.jpg', 'RussellMolimock2.jpg', 'JaiberRamirez.jpg', 'JaiberRamirez2.jpg', 'RicardoGutierrez.jpg', 'DavidLatorre.jpg', 'DianaBoada10.jpg', 'JulienneTesoro7.jpg', 'PhuTruong5.jpg', 'SofiaCheung0.jpg', 'YesidGonzalez3.jpg', 'JavierCañon2.jpg', 'KennethCortesAguas2.jpg', 'JuanValencia.jpg', 'MariaCoyUlloa.jpg', 'LauraVasquezBernal1.jpg', 'DavidKwan1.jpg', 'SnehaDasaLakshminath.jpg', 'BrentJanski7.jpg', 'AndrewMaring2.jpg', 'KyleLitscher0.jpg', 'HaroldoVélezLora1.jpg', 'LeoByeon3.jpg', 'MarkHedgeland2.jpg', 'JuanDavidAmayaGaviria.jpg', 'HeimerRojas7.jpg', 'DianaBoada0.jpg', 'SofiaCheung7.jpg', 'ElaineYeung8.jpg', 'AnthonyLe4.jpg', 'ElaineYeung2.jpg', 'JaiberRamirez5.jpg', 'SamuelAlexanderFlorez6.jpg', 'FaizanKhan.jpg', 'MarkHedgeland.jpg', 'OlgaLucíaRodríguezToloza4.jpg', 'DennisPham5.jpg', 'MiaMorton2.jpg', 'DianaBoada4.jpg', 'DennisPham2.jpg', 'JavierCañon.jpg', 'MohamethSeck3.jpg', 'RodrigoCruz3.jpg', 'PhuTruong1.jpg', 'FeliciaHsieh.jpg', 'JavierCañon4.jpg', 'KyleLitscher2.jpg', 'LauraRoudge2.jpg', 'TimAssavarat4.jpg', 'FrancescaCantor3.jpg', 'OlgaLucíaRodríguezToloza5.jpg', 'HaroldoVélezLora3.jpg', 'KyleLitscher.jpg', 'MiaMorton0.jpg', 'OlgaLucíaRodríguezToloza3.jpg', 'BrentJanski.jpg', 'NgaLa7.jpg', 'AdamSedki1.jpg', 'SamuelAlexanderFlorez1.jpg', 'LauraRoudge1.jpg', 'SofiaCheung.jpg', 'GiovannyAlexanderRubioAlbornoz3.jpg', 'ElaineYeung1.jpg', 'HeimerRojas10.jpg', 'GiovannyAlexanderRubioAlbornoz2.jpg', 'ArthurDamm0.jpg', 'RicardoGutierrez0.jpg', 'MarkHedgeland0.jpg', 'JuanValencia0.jpg', 'LeineValente.jpg', 'LeoByeon1.jpg', 'LauraVasquezBernal3.jpg', 'AlishaSmith4.jpg', 'CarlosArias3.jpg', 'JulienneTesoro0.jpg', 'BrendanEliason0.jpg', 'RodrigoCruz6.jpg', 'MiaMorton3.jpg', 'RobertSebastianCastellanosRodriguez4.jpg', 'MiaMorton4.jpg', 'SamieAzad4.jpg', 'BrendanEliason.jpg', 'AndrewMaring0.jpg', 'LauraRoudge0.jpg', 'PhuTruong.jpg', 'MarianaPlazas.jpg', 'MariaCoyUlloa3.jpg', 'SofiaCheung6.jpg', 'YesidGonzalez0.jpg', 'SofiaCheung3.jpg', 'BrendanEliason2.jpg', 'AlishaSmith1.jpg', 'DianaBoada8.jpg', 'JulienneTesoro3.jpg', 'MarkHedgeland3.jpg', 'JuanValencia2.jpg', 'JaiberRamirez0.jpg', 'DianaBoada.jpg', 'FrancescaCantor1.jpg', 'RodrigoCruz10.jpg', 'SofiaCheung5.jpg', 'ChristianWilliams.jpg', 'DennisPham0.jpg', 'RodrigoCruz8.jpg', 'TimAssavarat0.jpg', 'DavidKwan4.jpg', 'RobertSebastianCastellanosRodriguez0.jpg', 'TuVo1.jpg', 'TuVo.jpg', 'SnehaDasaLakshminath4.jpg', 'TimAssavarat3.jpg', 'RussellMolimock.jpg', 'YesidGonzalez2.jpg', 'HeimerRojas8.jpg', 'XimenaCarolinaAndradeVargas3.jpg', 'RobertSebastianCastellanosRodriguez1.jpg', 'ElaineYeung0.jpg', 'SofiaCheung1.jpg', 'ArthurDamm6.jpg', 'BrentJanski5.jpg', 'YesidGonzalez.jpg', 'FrancescaCantor5.jpg', 'ArthurDamm5.jpg', 'KoomeMwiti1.jpg', 'GiovannyAlexanderRubioAlbornoz7.jpg', 'SamuelAlexanderFlorez0.jpg', 'KennethCortesAguas0.jpg', 'LeineValente3.jpg', 'LauraRoudge3.jpg', 'KyleLitscher1.jpg', 'GiovannyAlexanderRubioAlbornoz4.jpg', 'PhuTruong4.jpg', 'MariaCoyUlloa0.jpg', 'OlgaLucíaRodríguezToloza1.jpg', 'AdamSedki2.jpg', 'OlgaLucíaRodríguezToloza.jpg', 'JohnCook0.jpg', 'MohamethSeck5.jpg', 'SnehaDasaLakshminath1.jpg', 'MiaMorton.jpg', 'CarlosArias6.jpg', 'AdamSedki0.jpg', 'LauraRoudge4.jpg', 'JohnCook2.jpg', 'MohamethSeck4.jpg', 'AndrewMaring4.jpg', 'GiovannyAlexanderRubioAlbornoz6.jpg', 'DianaBoada2.jpg', 'ChristianWilliams3.jpg', 'JavierCañon6.jpg', 'DianaBoada7.jpg', 'MariaCoyUlloa2.jpg', 'ChristianWilliams1.jpg', 'FrancescaCantor.jpg', 'KyleLitscher4.jpg', 'NgaLa6.jpg', 'DiegoAndrésCastellanosRodríguez3.jpg', 'KennethCortesAguas3.jpg', 'GiovannyAlexanderRubioAlbornoz.jpg', 'DavidKwan0.jpg', 'ArthurDamm3.jpg', 'SamuelAlexanderFlorez8.jpg', 'OlgaLucíaRodríguezToloza2.jpg', 'DennisPham1.jpg', 'RicardoGutierrez2.jpg', 'RobertSebastianCastellanosRodriguez2.jpg', 'SofiaCheung4.jpg', 'SamieAzad.jpg', 'SnehaDasaLakshminath0.jpg', 'DiegoAndrésCastellanosRodríguez2.jpg', 'AnthonyLe.jpg', 'ChristianWilliams2.jpg', 'RicardoGutierrez7.jpg', 'NgaLa1.jpg', 'SamuelAlexanderFlorez5.jpg', 'DiegoAndrésCastellanosRodríguez.jpg', 'KirenSrinivasan4.jpg', 'CarlosArias1.jpg', 'BrittneyGoertzen.jpg', 'SamuelAlexanderFlorez4.jpg', 'ArthurDamm4.jpg', 'JaiberRamirez6.jpg', 'AndresMartinPeñaRivera.jpg', 'XimenaCarolinaAndradeVargas2.jpg', 'JavierCañon3.jpg', 'KirenSrinivasan2.jpg', 'DennisPham8.jpg', 'BrentJanski3.jpg', 'FrancescaCantor0.jpg', 'JulienneTesoro1.jpg', 'PhuTruong6.jpg', 'OmarMartínezBermúdez.jpg', 'HeimerRojas4.jpg', 'DianaBoada5.jpg', 'JavierCañon0.jpg', 'LauraVasquezBernal2.jpg', 'DennisPham4.jpg', 'HaroldoVélezLora2.jpg', 'ElaineYeung5.jpg', 'SamuelAlexanderFlorez10.jpg', 'JosefGoodyear.jpg', 'KirenSrinivasan1.jpg', 'MarkHedgeland4.jpg', 'MariaCoyUlloa1.jpg', 'MohamethSeck.jpg', 'GiovannyAlexanderRubioAlbornoz0.jpg', 'AnthonyLe3.jpg', 'LauraRoudge.jpg', 'LeoByeon2.jpg', 'KoomeMwiti2.jpg', 'JulienneTesoro.jpg', 'KirenSrinivasan3.jpg', 'JavierCañon1.jpg', 'ChristianWilliams5.jpg', 'RobertSebastianCastellanosRodriguez3.jpg', 'MariaCoyUlloa5.jpg', 'JulienneTesoro4.jpg', 'HeimerRojas0.jpg', 'NathanPetersen.jpg', 'JohnCook.jpg', 'ChristianWilliams4.jpg', 'AnthonyLe0.jpg', 'MohamethSeck1.jpg', 'TimAssavarat1.jpg']
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/save%20processed.png)

## Task7 - Generate Triplets
Also in utils.py, write a function def generate_triplets(images, filenames, triplet_names): that generates triplets:<br>
<br>
* images is a numpy.ndarray of shape (i, n, n, 3) containing the aligned images in the dataset
* * i is the number of images
* * n is the size of the aligned images
* filenames is a list of length i containing the corresponding filenames for images
* triplet_names is a list of length m of lists where each sublist contains the filenames of an anchor, positive, and negative image, respectively
* m is the number of triplets<br>

Returns: a list [A, P, N]
* A is a numpy.ndarray of shape (m, n, n, 3) containing the anchor images for all m triplets
* P is a numpy.ndarray of shape (m, n, n, 3) containing the positive images for all m triplets
* N is a numpy.ndarray of shape (m, n, n, 3) containing the negative images for all m triplets

```
ubuntu-xenial:0x0B-face_verification$ cat 7-main.py
#!/usr/bin/env python3

from utils import load_images, load_csv, generate_triplets
import numpy as np
import matplotlib.pyplot as plt

images, filenames = load_images('HBTNaligned', as_array=True)
triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
plt.subplot(1, 3, 1)
plt.title('Anchor:' + triplet_names[0][0])
plt.imshow(A[0])
plt.subplot(1, 3, 2)
plt.title('Positive:' + triplet_names[0][1])
plt.imshow(P[0])
plt.subplot(1, 3, 3)
plt.title('Negative:' + triplet_names[0][2])
plt.imshow(N[0])
plt.tight_layout()
plt.show()
ubuntu-xenial:0x0B-face_verification$ ./7-main.py
```
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/supervised_learning/data/other%20pictures/getting%20triplets.png)

## Task8 - Initialize Triplet Loss
Create a custom layer class TripletLoss that inherits from tensorflow.keras.layers.Layer:<br>
<br>
Create the class constructor `def __init__(self, alpha, **kwargs):`
* alpha is the alpha value used to calculate the triplet loss
* sets the public instance attribute alpha

```
ubuntu-xenial:0x0B-face_verification$ cat 8-main.py
#!/usr/bin/env python3

from triplet_loss import TripletLoss

print(TripletLoss.__bases__)
tl = TripletLoss(0.2)
print(tl.alpha)
print(sorted(tl.__dict__.keys()))

ubuntu-xenial:0x0B-face_verification$  ./8-main.py
(<class 'tensorflow.python.keras.engine.base_layer.Layer'>,)
0.2
['_activity_regularizer', '_call_convention', '_call_fn_args', '_callable_losses', '_compute_previous_mask', '_dtype', '_dynamic', '_eager_losses', '_expects_training_arg', '_inbound_nodes', '_initial_weights', '_layers', '_losses', '_metrics', '_metrics_tensors', '_mixed_precision_policy', '_name', '_non_trainable_weights', '_obj_reference_counts_dict', '_outbound_nodes', '_self_setattr_tracking', '_trainable_weights', '_updates', 'alpha', 'built', 'input_spec', 'stateful', 'supports_masking', 'trainable']
ubuntu-xenial:0x0B-face_verification$
```
## Task9 - Calculate Triplet Loss 
Update the class TripletLoss:<br>
<br>
Create the public instance method `def triplet_loss(self, inputs):`
* inputs is a list containing the anchor, positive and negative output tensors from the last layer of the model, respectively<br>

Returns: a tensor containing the triplet loss values

```
ubuntu-xenial:0x0B-face_verification$ cat 9-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from triplet_loss import TripletLoss

np.random.seed(0)
tl = TripletLoss(0.2)
A = np.random.uniform(0, 1, (2, 128))
P = np.random.uniform(0, 1, (2, 128))
N = np.random.uniform(0, 1, (2, 128))

with tf.Session() as sess:
    loss = tl.triplet_loss([A, P, N])
    print(type(loss))
    print(loss.eval())
ubuntu-xenial:0x0B-face_verification$ ./9-main.py
<class 'tensorflow.python.framework.ops.Tensor'>
[0.         3.31159856]
ubuntu-xenial:0x0B-face_verification$
```

## Task10 - Call Triplet Loss
Update the class TripletLoss:<br>
<br>
Create the public instance method `def call(self, inputs):`
* inputs is a list containing the anchor, positive, and negative output tensors from the last layer of the model, respectively
* adds the triplet loss to the graph<br>

Returns: the triplet loss tensor

```
ubuntu-xenial:0x0B-face_verification$ cat 10-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from triplet_loss import TripletLoss

np.random.seed(0)
tl = TripletLoss(0.2)
A = np.random.uniform(0, 1, (2, 128))
P = np.random.uniform(0, 1, (2, 128))
N = np.random.uniform(0, 1, (2, 128))

inputs = [tf.keras.Input((128,)), tf.keras.Input((128,)), tf.keras.Input((128,))]
output = tl(inputs)

model = tf.keras.models.Model(inputs, output)
print(output)
print(tl._losses)
print(model.losses)
print(model.predict([A, P, N]))

ubuntu-xenial:0x0B-face_verification$ ./10-main.py
Tensor("triplet_loss/Maximum:0", shape=(?,), dtype=float32)
[<tf.Tensor 'triplet_loss/Maximum:0' shape=(?,) dtype=float32>]
[<tf.Tensor 'triplet_loss/Maximum:0' shape=(?,) dtype=float32>]
[0.       3.311599]
ubuntu-xenial:0x0B-face_verification$
```

## Task11 - Initialize Train Model
Create the class TrainModel that trains a model for face verification using triplet loss:<br>
<br>
Create the class constructor `def __init__(self, model_path, alpha):`
* model_path is the path to the base face verification embedding model
* loads the model using with tf.keras.utils.CustomObjectScope({'tf': tf}):
* saves this model as the public instance method base_model
* alpha is the alpha to use for the triplet loss calculation
* Creates a new model:
* inputs: [A, P, N]
* * A is a numpy.ndarray of shape (m, n, n, 3)containing the aligned anchor images
* * P is a numpy.ndarray of shape (m, n, n, 3) containing the aligned positive images
* * N is a numpy.ndarray of shape (m, n, n, 3)containing the aligned negative images
* * m is the number of images
* * n is the size of the aligned images
* outputs: the triplet losses of base_model
* compiles the model with:
* Adam optimization
* no additional losses
* save this model as the public instance attribute training_model
* you can use from triplet_loss import TripletLoss


```
ubuntu-xenial:0x0B-face_verification$ cat 11-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
images = images.astype('float32') / 255

triplet_names = load_csv('FVTriplets.csv')
triplets = generate_triplets(images, filenames, triplet_names)

tm = TrainModel('models/face_verification.h5', 0.2)
tm.training_model.summary()
losses = tm.training_model.predict(triplets, batch_size=1)
print(losses.shape)
print(np.mean(losses))

ubuntu-xenial:0x0B-face_verification$ ./11-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:Output "triplet_loss_layer" missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to "triplet_loss_layer".
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 96, 96, 3)    0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 96, 96, 3)    0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 96, 96, 3)    0                                            
__________________________________________________________________________________________________
model (Model)                   (None, 128)          3743280     input_1[0][0]                    
                                                                 input_2[0][0]                    
                                                                 input_3[0][0]                    
__________________________________________________________________________________________________
triplet_loss_layer (TripletLoss (None,)              0           model[1][0]                      
                                                                 model[2][0]                      
                                                                 model[3][0]                      
==================================================================================================
Total params: 3,743,280
Trainable params: 3,733,968
Non-trainable params: 9,312
__________________________________________________________________________________________________
5122/5122 [==============================] - 21s 4ms/step
(5122,)
0.08501873 #this number depends of your training
ubuntu-xenial:0x0B-face_verification$
```

## Task12 - Train
Update the class TrainModel:<br>
<br>
Create the public instance method `def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3, verbose=True):` that trains self.training_model:
* triplets is a list of numpy.ndarrayscontaining the inputs to self.training_model
* epochs is the number of epochs to train for
* batch_size is the batch size for training
* validation_split is the validation split for training
* verbose is a boolean that sets the verbosity mode<br>

Returns: the History output from the training

```
ubuntu-xenial:0x0B-face_verification$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
images = images.astype('float32') / 255

triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
triplets = [A[:-2], P[:-2], N[:-2]] # to make all batches divisible by 32

tm = TrainModel('models/face_verification.h5', 0.2)
history = tm.train(triplets, epochs=1)
print(history.history)

ubuntu-xenial:0x0B-face_verification$ ./12-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:Output "triplet_loss_layer" missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to "triplet_loss_layer".
Train on 3584 samples, validate on 1536 samples
Epoch 1/1
3584/3584 [==============================] - 84s 24ms/step - loss: 0.0066 - val_loss: 0.1407
{'loss': [array([4.9168253e-03, 8.2771722e-03, 5.3734081e-03, 1.0293166e-02,
       1.4802018e-02, 3.5686307e-03, 7.1085799e-03, 8.5692722e-03,
       0.0000000e+00, 8.8522984e-03, 4.3967427e-03, 1.5823873e-02,
       2.7977589e-03, 3.4197222e-03, 1.3445925e-03, 7.0106550e-03,
       4.0564910e-03, 2.2684508e-03, 1.2449501e-02, 7.9540377e-03,
       4.5845318e-03, 6.3860491e-03, 5.3574713e-03, 5.4465644e-03,
       2.8503905e-03, 5.1279212e-03, 0.0000000e+00, 6.7495825e-03,
       9.8483777e-03, 7.4272775e-03, 2.4280086e-02, 2.0542448e-05],
      dtype=float32)], 'val_loss': [array([0.17247383, 0.18618198, 0.15363973, 0.1578978 , 0.1369457 ,
       0.16085257, 0.16726387, 0.15662782, 0.1651832 , 0.16623686,
       0.13304965, 0.11869448, 0.11358245, 0.14346656, 0.1001007 ,
       0.17649145, 0.1274278 , 0.11775222, 0.10644331, 0.10001153,
       0.11342994, 0.10343826, 0.12853147, 0.14112641, 0.13591947,
       0.14552793, 0.12288004, 0.12684627, 0.12555532, 0.14132085,
       0.17476451, 0.18300104], dtype=float32)]}
ubuntu-xenial:0x0B-face_verification$
```

## Task13 - Save
Update the class TrainModel:<br<
<br>
Create the public instance method `def save(self, save_path):` that saves the base embedding model:
* save_path is the path to save the model
* Returns: the saved model

```
ubuntu-xenial:0x0B-face_verification$ cat 13-main.py
#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('HBTNaligned', as_array=True)
images = images.astype('float32') / 255

triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
triplets = [A[:-2], P[:-2], N[:-2]] 

tm = TrainModel('models/face_verification.h5', 0.2)
tm.train(triplets, epochs=1)
base_model = tm.save('models/trained_fv.h5')
print(base_model is tm.base_model)
print(os.listdir('models'))

ubuntu-xenial:0x0B-face_verification$ ./13-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:Output "triplet_loss_layer" missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to "triplet_loss_layer".
Train on 3584 samples, validate on 1536 samples
Epoch 1/1
3584/3584 [==============================] - 69s 19ms/step - loss: 0.0053 - val_loss: 0.1373
True
['face_verification.h5', 'trained_fv.h5', 'landmarks.dat']
ubuntu-xenial:0x0B-face_verification$
```

## Task14 - Calculate Metrics 
Update the class TrainModel:<br>
<br>
static method `def f1_score(y_true, y_pred):` that calculates the F1 score of predictions
* y_true - a numpy.ndarray of shape (m,) containing the correct labels
* m is the number of examples
* y_pred- a numpy.ndarray of shape (m,) containing the predicted labels<br>

Returns: The f1 score<br>
<br>

static method `def accuracy(y_true, y_pred):`
* y_true - a numpy.ndarray of shape (m,) containing the correct labels
* m is the number of examples
* y_pred- a numpy.ndarray of shape (m,) containing the predicted labels<br>

Returns: the accuracy

```
ubuntu-xenial:0x0B-face_verification$ cat 14-main.py
#!/usr/bin/env python3

import numpy as np
from train_model import TrainModel

tm = TrainModel('models/face_verification.h5', 0.2)

np.random.seed(0)
y_true = np.random.randint(0, 2, 10)
y_pred = np.random.randint(0, 2, 10)
print(tm.f1_score(y_true, y_pred))
print(tm.accuracy(y_true, y_pred))

ubuntu-xenial:0x0B-face_verification$ ./14-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:Output "triplet_loss_layer" missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to "triplet_loss_layer".
0.18181818181818182
0.1
ubuntu-xenial:0x0B-face_verification$
```
## Task15 -Tau
Update the class TrainModel:<br>
<br>
public instance method `def best_tau(self, images, identities, thresholds):` that calculates the best tau to use for a maximal F1 score
* images - a numpy.ndarray of shape (m, n, n, 3) containing the aligned images for testing
* * m is the number of images
* * n is the size of the images
* identities - a list containing the identities of each image in images
* thresholds - a 1D numpy.ndarray of distance thresholds (tau) to test<br>

Returns: (tau, f1, acc)
* tau- the optimal threshold to maximize F1 score
* f1 - the maximal F1 score
* acc - the accuracy associated with the maximal F1 score

```
ubuntu-xenial:0x0B-face_verification$ cat 15-main.py
#!/usr/bin/env python3

import numpy as np
import re
from train_model import TrainModel
from utils import load_images


images, filenames = load_images('HBTNaligned', as_array=True)
images = images.astype('float32') / 255

identities = [re.sub('[0-9]', '', f[:-4]) for f in filenames]
print(set(identities))
thresholds = np.linspace(0.05, 1, 96)
tm = TrainModel('models/face_verification.h5', 0.2)
tau, f1, acc = tm.best_tau(images, identities, thresholds)
print(tau)
print(f1)
print(acc)
ubuntu-xenial:0x0B-face_verification$ ./15-main.py
{'AllisonWeiner', 'DianaBoada', 'SamuelAlexanderFlorez', 'DiegoAndrésCastellanosRodríguez', 'XimenaCarolinaAndradeVargas', 'OmarMartínezBermúdez', 'HaroldoVélezLora', 'JohnCook', 'DavidLatorre', 'AlishaSmith', 'TimAssavarat', 'ElaineYeung', 'JuanValencia', 'JavierCañon', 'AnthonyLe', 'FrancescaCantor', 'DavidKwan', 'NathanPetersen', 'DennisPham', 'LeoByeon', 'MariaCoyUlloa', 'MarianaPlazas', 'RyanHudson', 'FeliciaHsieh', 'JosefGoodyear', 'BrentJanski', 'LauraVasquezBernal', 'SnehaDasaLakshminath', 'KoomeMwiti', 'JuanDavidAmayaGaviria', 'KennethCortesAguas', 'KyleLitscher', 'MiaMorton', 'TuVo', 'PhuTruong', 'RussellMolimock', 'OlgaLucíaRodríguezToloza', 'HongtuHuang', 'CarlosArias', 'ArthurDamm', 'MarkHedgeland', 'SofiaCheung', 'JulienneTesoro', 'FaizanKhan', 'ChristianWilliams', 'KirenSrinivasan', 'AndresMartinPeñaRivera', 'BrittneyGoertzen', 'LauraRoudge', 'MohamethSeck', 'RicardoGutierrez', 'AndrewMaring', 'RodrigoCruz', 'BrendanEliason', 'RobertSebastianCastellanosRodriguez', 'AdamSedki', 'JaiberRamirez', 'GiovannyAlexanderRubioAlbornoz', 'NgaLa', 'HeimerRojas', 'SamieAzad', 'YesidGonzalez', 'LeineValente'}
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:Output "triplet_loss_layer" missing from loss dictionary. We assume this was done on purpose. The fit and evaluate APIs will not be expecting any data to be passed to "triplet_loss_layer".
# this values also can change, it depends of your training
0.44
0.5446546414430268
0.9859983766233766
ubuntu-xenial:0x0B-face_verification$
```

## Task16 - Initialize Face Verification
Create the class FaceVerification:<br>
<br>
class constructor `def __init__(self, model_path, database, identities):`
* model_path is the path to where the face verification embedding model is stored
* you will need to use with tf.keras.utils.CustomObjectScope({'tf': tf}): to load the model
* database is a numpy.ndarray of shape (d, e) containing all the face embeddings in the database
* * d is the number of images in the database
* * e is the dimensionality of the embedding
* identities is a list of length d containing the identities corresponding to the embeddings in database
* Sets the public instance attributes model, database and identities

```
ubuntu-xenial:0x0B-face_verification$ cat 16-main.py
#!/usr/bin/env python3                                                                                                                                                      

import numpy as np
from verification import FaceVerification

np.random.seed(0)
database = np.random.randn(5, 128)
identities = ['Holberton', 'school', 'is', 'the', 'best!']
fv = FaceVerification('models/trained_fv.h5', database, identities)
fv.model.summary()
print(fv.database)
print(fv.identities)
ubuntu-xenial:0x0B-face_verification$ ./16-main.py
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 96, 96, 3)    0                                            
__________________________________________________________________________________________________
zero_padding2d (ZeroPadding2D)  (None, 102, 102, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 48, 48, 64)   9472        zero_padding2d[0][0]             
__________________________________________________________________________________________________
bn1 (BatchNormalization)        (None, 48, 48, 64)   256         conv1[0][0]                      
__________________________________________________________________________________________________
activation (Activation)         (None, 48, 48, 64)   0           bn1[0][0]                        
__________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D (None, 50, 50, 64)   0           activation[0][0]                 
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 24, 24, 64)   0           zero_padding2d_1[0][0]           
__________________________________________________________________________________________________
lrn_1 (Lambda)                  (None, 24, 24, 64)   0           max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2 (Conv2D)                  (None, 24, 24, 64)   4160        lrn_1[0][0]                      
__________________________________________________________________________________________________
bn2 (BatchNormalization)        (None, 24, 24, 64)   256         conv2[0][0]                      
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 24, 24, 64)   0           bn2[0][0]                        
__________________________________________________________________________________________________
zero_padding2d_2 (ZeroPadding2D (None, 26, 26, 64)   0           activation_1[0][0]               
__________________________________________________________________________________________________
conv3 (Conv2D)                  (None, 24, 24, 192)  110784      zero_padding2d_2[0][0]           
__________________________________________________________________________________________________
bn3 (BatchNormalization)        (None, 24, 24, 192)  768         conv3[0][0]                      
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 24, 24, 192)  0           bn3[0][0]                        
__________________________________________________________________________________________________
lrn_2 (Lambda)                  (None, 24, 24, 192)  0           activation_2[0][0]               
__________________________________________________________________________________________________
zero_padding2d_3 (ZeroPadding2D (None, 26, 26, 192)  0           lrn_2[0][0]                      
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 12, 12, 192)  0           zero_padding2d_3[0][0]           
__________________________________________________________________________________________________
inception_3a_3x3_conv1 (Conv2D) (None, 12, 12, 96)   18528       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
inception_3a_5x5_conv1 (Conv2D) (None, 12, 12, 16)   3088        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
inception_3a_3x3_bn1 (BatchNorm (None, 12, 12, 96)   384         inception_3a_3x3_conv1[0][0]     
__________________________________________________________________________________________________
inception_3a_5x5_bn1 (BatchNorm (None, 12, 12, 16)   64          inception_3a_5x5_conv1[0][0]     
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 12, 12, 96)   0           inception_3a_3x3_bn1[0][0]       
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 12, 12, 16)   0           inception_3a_5x5_bn1[0][0]       
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 5, 5, 192)    0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
zero_padding2d_4 (ZeroPadding2D (None, 14, 14, 96)   0           activation_3[0][0]               
__________________________________________________________________________________________________
zero_padding2d_5 (ZeroPadding2D (None, 16, 16, 16)   0           activation_5[0][0]               
__________________________________________________________________________________________________
inception_3a_pool_conv (Conv2D) (None, 5, 5, 32)     6176        max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
inception_3a_3x3_conv2 (Conv2D) (None, 12, 12, 128)  110720      zero_padding2d_4[0][0]           
__________________________________________________________________________________________________
inception_3a_5x5_conv2 (Conv2D) (None, 12, 12, 32)   12832       zero_padding2d_5[0][0]           
__________________________________________________________________________________________________
inception_3a_pool_bn (BatchNorm (None, 5, 5, 32)     128         inception_3a_pool_conv[0][0]     
__________________________________________________________________________________________________
inception_3a_1x1_conv (Conv2D)  (None, 12, 12, 64)   12352       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
inception_3a_3x3_bn2 (BatchNorm (None, 12, 12, 128)  512         inception_3a_3x3_conv2[0][0]     
__________________________________________________________________________________________________
inception_3a_5x5_bn2 (BatchNorm (None, 12, 12, 32)   128         inception_3a_5x5_conv2[0][0]     
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 5, 5, 32)     0           inception_3a_pool_bn[0][0]       
__________________________________________________________________________________________________
inception_3a_1x1_bn (BatchNorma (None, 12, 12, 64)   256         inception_3a_1x1_conv[0][0]      
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 12, 12, 128)  0           inception_3a_3x3_bn2[0][0]       
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 12, 12, 32)   0           inception_3a_5x5_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_6 (ZeroPadding2D (None, 12, 12, 32)   0           activation_7[0][0]               
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 12, 12, 64)   0           inception_3a_1x1_bn[0][0]        
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 12, 12, 256)  0           activation_4[0][0]               
                                                                 activation_6[0][0]               
                                                                 zero_padding2d_6[0][0]           
                                                                 activation_8[0][0]               
__________________________________________________________________________________________________
inception_3b_3x3_conv1 (Conv2D) (None, 12, 12, 96)   24672       concatenate[0][0]                
__________________________________________________________________________________________________
inception_3b_5x5_conv1 (Conv2D) (None, 12, 12, 32)   8224        concatenate[0][0]                
__________________________________________________________________________________________________
inception_3b_3x3_bn1 (BatchNorm (None, 12, 12, 96)   384         inception_3b_3x3_conv1[0][0]     
__________________________________________________________________________________________________
inception_3b_5x5_bn1 (BatchNorm (None, 12, 12, 32)   128         inception_3b_5x5_conv1[0][0]     
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 12, 12, 96)   0           inception_3b_3x3_bn1[0][0]       
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 12, 12, 32)   0           inception_3b_5x5_bn1[0][0]       
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 4, 4, 256)    0           concatenate[0][0]                
__________________________________________________________________________________________________
zero_padding2d_7 (ZeroPadding2D (None, 14, 14, 96)   0           activation_9[0][0]               
__________________________________________________________________________________________________
zero_padding2d_8 (ZeroPadding2D (None, 16, 16, 32)   0           activation_11[0][0]              
__________________________________________________________________________________________________
inception_3b_pool_conv (Conv2D) (None, 4, 4, 64)     16448       average_pooling2d[0][0]          
__________________________________________________________________________________________________
inception_3b_3x3_conv2 (Conv2D) (None, 12, 12, 128)  110720      zero_padding2d_7[0][0]           
__________________________________________________________________________________________________
inception_3b_5x5_conv2 (Conv2D) (None, 12, 12, 64)   51264       zero_padding2d_8[0][0]           
__________________________________________________________________________________________________
inception_3b_pool_bn (BatchNorm (None, 4, 4, 64)     256         inception_3b_pool_conv[0][0]     
__________________________________________________________________________________________________
inception_3b_1x1_conv (Conv2D)  (None, 12, 12, 64)   16448       concatenate[0][0]                
__________________________________________________________________________________________________
inception_3b_3x3_bn2 (BatchNorm (None, 12, 12, 128)  512         inception_3b_3x3_conv2[0][0]     
__________________________________________________________________________________________________
inception_3b_5x5_bn2 (BatchNorm (None, 12, 12, 64)   256         inception_3b_5x5_conv2[0][0]     
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 4, 4, 64)     0           inception_3b_pool_bn[0][0]       
__________________________________________________________________________________________________
inception_3b_1x1_bn (BatchNorma (None, 12, 12, 64)   256         inception_3b_1x1_conv[0][0]      
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 12, 12, 128)  0           inception_3b_3x3_bn2[0][0]       
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 12, 12, 64)   0           inception_3b_5x5_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_9 (ZeroPadding2D (None, 12, 12, 64)   0           activation_13[0][0]              
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 12, 12, 64)   0           inception_3b_1x1_bn[0][0]        
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 12, 12, 320)  0           activation_10[0][0]              
                                                                 activation_12[0][0]              
                                                                 zero_padding2d_9[0][0]           
                                                                 activation_14[0][0]              
__________________________________________________________________________________________________
inception_3c_3x3_conv1 (Conv2D) (None, 12, 12, 128)  41088       concatenate_1[0][0]              
__________________________________________________________________________________________________
inception_3c_5x5_conv1 (Conv2D) (None, 12, 12, 32)   10272       concatenate_1[0][0]              
__________________________________________________________________________________________________
inception_3c_3x3_bn1 (BatchNorm (None, 12, 12, 128)  512         inception_3c_3x3_conv1[0][0]     
__________________________________________________________________________________________________
inception_3c_5x5_bn1 (BatchNorm (None, 12, 12, 32)   128         inception_3c_5x5_conv1[0][0]     
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 12, 12, 128)  0           inception_3c_3x3_bn1[0][0]       
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 12, 12, 32)   0           inception_3c_5x5_bn1[0][0]       
__________________________________________________________________________________________________
zero_padding2d_10 (ZeroPadding2 (None, 14, 14, 128)  0           activation_15[0][0]              
__________________________________________________________________________________________________
zero_padding2d_11 (ZeroPadding2 (None, 16, 16, 32)   0           activation_17[0][0]              
__________________________________________________________________________________________________
inception_3c_3x3_conv2 (Conv2D) (None, 6, 6, 256)    295168      zero_padding2d_10[0][0]          
__________________________________________________________________________________________________
inception_3c_5x5_conv2 (Conv2D) (None, 6, 6, 64)     51264       zero_padding2d_11[0][0]          
__________________________________________________________________________________________________
inception_3c_3x3_bn2 (BatchNorm (None, 6, 6, 256)    1024        inception_3c_3x3_conv2[0][0]     
__________________________________________________________________________________________________
inception_3c_5x5_bn2 (BatchNorm (None, 6, 6, 64)     256         inception_3c_5x5_conv2[0][0]     
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 5, 5, 320)    0           concatenate_1[0][0]              
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 6, 6, 256)    0           inception_3c_3x3_bn2[0][0]       
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 6, 6, 64)     0           inception_3c_5x5_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_12 (ZeroPadding2 (None, 6, 6, 320)    0           max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 6, 6, 640)    0           activation_16[0][0]              
                                                                 activation_18[0][0]              
                                                                 zero_padding2d_12[0][0]          
__________________________________________________________________________________________________
inception_4a_3x3_conv1 (Conv2D) (None, 6, 6, 96)     61536       concatenate_2[0][0]              
__________________________________________________________________________________________________
inception_4a_5x5_conv1 (Conv2D) (None, 6, 6, 32)     20512       concatenate_2[0][0]              
__________________________________________________________________________________________________
inception_4a_3x3_bn1 (BatchNorm (None, 6, 6, 96)     384         inception_4a_3x3_conv1[0][0]     
__________________________________________________________________________________________________
inception_4a_5x5_bn1 (BatchNorm (None, 6, 6, 32)     128         inception_4a_5x5_conv1[0][0]     
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 6, 6, 96)     0           inception_4a_3x3_bn1[0][0]       
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 6, 6, 32)     0           inception_4a_5x5_bn1[0][0]       
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 2, 2, 640)    0           concatenate_2[0][0]              
__________________________________________________________________________________________________
zero_padding2d_13 (ZeroPadding2 (None, 8, 8, 96)     0           activation_19[0][0]              
__________________________________________________________________________________________________
zero_padding2d_14 (ZeroPadding2 (None, 10, 10, 32)   0           activation_21[0][0]              
__________________________________________________________________________________________________
inception_4a_pool_conv (Conv2D) (None, 2, 2, 128)    82048       average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
inception_4a_3x3_conv2 (Conv2D) (None, 6, 6, 192)    166080      zero_padding2d_13[0][0]          
__________________________________________________________________________________________________
inception_4a_5x5_conv2 (Conv2D) (None, 6, 6, 64)     51264       zero_padding2d_14[0][0]          
__________________________________________________________________________________________________
inception_4a_pool_bn (BatchNorm (None, 2, 2, 128)    512         inception_4a_pool_conv[0][0]     
__________________________________________________________________________________________________
inception_4a_1x1_conv (Conv2D)  (None, 6, 6, 256)    164096      concatenate_2[0][0]              
__________________________________________________________________________________________________
inception_4a_3x3_bn2 (BatchNorm (None, 6, 6, 192)    768         inception_4a_3x3_conv2[0][0]     
__________________________________________________________________________________________________
inception_4a_5x5_bn2 (BatchNorm (None, 6, 6, 64)     256         inception_4a_5x5_conv2[0][0]     
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 2, 2, 128)    0           inception_4a_pool_bn[0][0]       
__________________________________________________________________________________________________
inception_4a_1x1_bn (BatchNorma (None, 6, 6, 256)    1024        inception_4a_1x1_conv[0][0]      
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 6, 6, 192)    0           inception_4a_3x3_bn2[0][0]       
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 6, 6, 64)     0           inception_4a_5x5_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_15 (ZeroPadding2 (None, 6, 6, 128)    0           activation_23[0][0]              
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 6, 6, 256)    0           inception_4a_1x1_bn[0][0]        
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 6, 6, 640)    0           activation_20[0][0]              
                                                                 activation_22[0][0]              
                                                                 zero_padding2d_15[0][0]          
                                                                 activation_24[0][0]              
__________________________________________________________________________________________________
inception_4e_3x3_conv1 (Conv2D) (None, 6, 6, 160)    102560      concatenate_3[0][0]              
__________________________________________________________________________________________________
inception_4e_5x5_conv1 (Conv2D) (None, 6, 6, 64)     41024       concatenate_3[0][0]              
__________________________________________________________________________________________________
inception_4e_3x3_bn1 (BatchNorm (None, 6, 6, 160)    640         inception_4e_3x3_conv1[0][0]     
__________________________________________________________________________________________________
inception_4e_5x5_bn1 (BatchNorm (None, 6, 6, 64)     256         inception_4e_5x5_conv1[0][0]     
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 6, 6, 160)    0           inception_4e_3x3_bn1[0][0]       
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 6, 6, 64)     0           inception_4e_5x5_bn1[0][0]       
__________________________________________________________________________________________________
zero_padding2d_16 (ZeroPadding2 (None, 8, 8, 160)    0           activation_25[0][0]              
__________________________________________________________________________________________________
zero_padding2d_17 (ZeroPadding2 (None, 10, 10, 64)   0           activation_27[0][0]              
__________________________________________________________________________________________________
inception_4e_3x3_conv2 (Conv2D) (None, 3, 3, 256)    368896      zero_padding2d_16[0][0]          
__________________________________________________________________________________________________
inception_4e_5x5_conv2 (Conv2D) (None, 3, 3, 128)    204928      zero_padding2d_17[0][0]          
__________________________________________________________________________________________________
inception_4e_3x3_bn2 (BatchNorm (None, 3, 3, 256)    1024        inception_4e_3x3_conv2[0][0]     
__________________________________________________________________________________________________
inception_4e_5x5_bn2 (BatchNorm (None, 3, 3, 128)    512         inception_4e_5x5_conv2[0][0]     
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 2, 640)    0           concatenate_3[0][0]              
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 3, 3, 256)    0           inception_4e_3x3_bn2[0][0]       
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 3, 3, 128)    0           inception_4e_5x5_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_18 (ZeroPadding2 (None, 3, 3, 640)    0           max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 3, 3, 1024)   0           activation_26[0][0]              
                                                                 activation_28[0][0]              
                                                                 zero_padding2d_18[0][0]          
__________________________________________________________________________________________________
inception_5a_3x3_conv1 (Conv2D) (None, 3, 3, 96)     98400       concatenate_4[0][0]              
__________________________________________________________________________________________________
inception_5a_3x3_bn1 (BatchNorm (None, 3, 3, 96)     384         inception_5a_3x3_conv1[0][0]     
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 3, 3, 96)     0           inception_5a_3x3_bn1[0][0]       
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 1, 1, 1024)   0           concatenate_4[0][0]              
__________________________________________________________________________________________________
zero_padding2d_19 (ZeroPadding2 (None, 5, 5, 96)     0           activation_29[0][0]              
__________________________________________________________________________________________________
inception_5a_pool_conv (Conv2D) (None, 1, 1, 96)     98400       average_pooling2d_2[0][0]        
__________________________________________________________________________________________________
inception_5a_3x3_conv2 (Conv2D) (None, 3, 3, 384)    332160      zero_padding2d_19[0][0]          
__________________________________________________________________________________________________
inception_5a_pool_bn (BatchNorm (None, 1, 1, 96)     384         inception_5a_pool_conv[0][0]     
__________________________________________________________________________________________________
inception_5a_1x1_conv (Conv2D)  (None, 3, 3, 256)    262400      concatenate_4[0][0]              
__________________________________________________________________________________________________
inception_5a_3x3_bn2 (BatchNorm (None, 3, 3, 384)    1536        inception_5a_3x3_conv2[0][0]     
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 1, 1, 96)     0           inception_5a_pool_bn[0][0]       
__________________________________________________________________________________________________
inception_5a_1x1_bn (BatchNorma (None, 3, 3, 256)    1024        inception_5a_1x1_conv[0][0]      
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 3, 3, 384)    0           inception_5a_3x3_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_20 (ZeroPadding2 (None, 3, 3, 96)     0           activation_31[0][0]              
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 3, 3, 256)    0           inception_5a_1x1_bn[0][0]        
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 3, 3, 736)    0           activation_30[0][0]              
                                                                 zero_padding2d_20[0][0]          
                                                                 activation_32[0][0]              
__________________________________________________________________________________________________
inception_5b_3x3_conv1 (Conv2D) (None, 3, 3, 96)     70752       concatenate_5[0][0]              
__________________________________________________________________________________________________
inception_5b_3x3_bn1 (BatchNorm (None, 3, 3, 96)     384         inception_5b_3x3_conv1[0][0]     
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 3, 3, 96)     0           inception_5b_3x3_bn1[0][0]       
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 1, 1, 736)    0           concatenate_5[0][0]              
__________________________________________________________________________________________________
zero_padding2d_21 (ZeroPadding2 (None, 5, 5, 96)     0           activation_33[0][0]              
__________________________________________________________________________________________________
inception_5b_pool_conv (Conv2D) (None, 1, 1, 96)     70752       max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
inception_5b_3x3_conv2 (Conv2D) (None, 3, 3, 384)    332160      zero_padding2d_21[0][0]          
__________________________________________________________________________________________________
inception_5b_pool_bn (BatchNorm (None, 1, 1, 96)     384         inception_5b_pool_conv[0][0]     
__________________________________________________________________________________________________
inception_5b_1x1_conv (Conv2D)  (None, 3, 3, 256)    188672      concatenate_5[0][0]              
__________________________________________________________________________________________________
inception_5b_3x3_bn2 (BatchNorm (None, 3, 3, 384)    1536        inception_5b_3x3_conv2[0][0]     
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 1, 1, 96)     0           inception_5b_pool_bn[0][0]       
__________________________________________________________________________________________________
inception_5b_1x1_bn (BatchNorma (None, 3, 3, 256)    1024        inception_5b_1x1_conv[0][0]      
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 3, 3, 384)    0           inception_5b_3x3_bn2[0][0]       
__________________________________________________________________________________________________
zero_padding2d_22 (ZeroPadding2 (None, 3, 3, 96)     0           activation_35[0][0]              
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 3, 3, 256)    0           inception_5b_1x1_bn[0][0]        
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 3, 3, 736)    0           activation_34[0][0]              
                                                                 zero_padding2d_22[0][0]          
                                                                 activation_36[0][0]              
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 1, 1, 736)    0           concatenate_6[0][0]              
__________________________________________________________________________________________________
flatten (Flatten)               (None, 736)          0           average_pooling2d_3[0][0]        
__________________________________________________________________________________________________
dense_layer (Dense)             (None, 128)          94336       flatten[0][0]                    
__________________________________________________________________________________________________
norm_layer (Lambda)             (None, 128)          0           dense_layer[0][0]                
==================================================================================================
Total params: 3,743,280
Trainable params: 3,733,968
Non-trainable params: 9,312
__________________________________________________________________________________________________
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01
  -1.03218852e-01  4.10598502e-01  1.44043571e-01  1.45427351e+00
   7.61037725e-01  1.21675016e-01  4.43863233e-01  3.33674327e-01
   1.49407907e+00 -2.05158264e-01  3.13067702e-01 -8.54095739e-01
  -2.55298982e+00  6.53618595e-01  8.64436199e-01 -7.42165020e-01
   2.26975462e+00 -1.45436567e+00  4.57585173e-02 -1.87183850e-01
   1.53277921e+00  1.46935877e+00  1.54947426e-01  3.78162520e-01
  -8.87785748e-01 -1.98079647e+00 -3.47912149e-01  1.56348969e-01
   1.23029068e+00  1.20237985e+00 -3.87326817e-01 -3.02302751e-01
  -1.04855297e+00 -1.42001794e+00 -1.70627019e+00  1.95077540e+00
  -5.09652182e-01 -4.38074302e-01 -1.25279536e+00  7.77490356e-01
  -1.61389785e+00 -2.12740280e-01 -8.95466561e-01  3.86902498e-01
  -5.10805138e-01 -1.18063218e+00 -2.81822283e-02  4.28331871e-01
   6.65172224e-02  3.02471898e-01 -6.34322094e-01 -3.62741166e-01
  -6.72460448e-01 -3.59553162e-01 -8.13146282e-01 -1.72628260e+00
   1.77426142e-01 -4.01780936e-01 -1.63019835e+00  4.62782256e-01
  -9.07298364e-01  5.19453958e-02  7.29090562e-01  1.28982911e-01
   1.13940068e+00 -1.23482582e+00  4.02341641e-01 -6.84810091e-01
  -8.70797149e-01 -5.78849665e-01 -3.11552532e-01  5.61653422e-02
  -1.16514984e+00  9.00826487e-01  4.65662440e-01 -1.53624369e+00
   1.48825219e+00  1.89588918e+00  1.17877957e+00 -1.79924836e-01
  -1.07075262e+00  1.05445173e+00 -4.03176947e-01  1.22244507e+00
   2.08274978e-01  9.76639036e-01  3.56366397e-01  7.06573168e-01
   1.05000207e-02  1.78587049e+00  1.26912093e-01  4.01989363e-01
   1.88315070e+00 -1.34775906e+00 -1.27048500e+00  9.69396708e-01
  -1.17312341e+00  1.94362119e+00 -4.13618981e-01 -7.47454811e-01
   1.92294203e+00  1.48051479e+00  1.86755896e+00  9.06044658e-01
  -8.61225685e-01  1.91006495e+00 -2.68003371e-01  8.02456396e-01
   9.47251968e-01 -1.55010093e-01  6.14079370e-01  9.22206672e-01
   3.76425531e-01 -1.09940079e+00  2.98238174e-01  1.32638590e+00
  -6.94567860e-01 -1.49634540e-01 -4.35153552e-01  1.84926373e+00]
 [ 6.72294757e-01  4.07461836e-01 -7.69916074e-01  5.39249191e-01
  -6.74332661e-01  3.18305583e-02 -6.35846078e-01  6.76433295e-01
   5.76590817e-01 -2.08298756e-01  3.96006713e-01 -1.09306151e+00
  -1.49125759e+00  4.39391701e-01  1.66673495e-01  6.35031437e-01
   2.38314477e+00  9.44479487e-01 -9.12822225e-01  1.11701629e+00
  -1.31590741e+00 -4.61584605e-01 -6.82416053e-02  1.71334272e+00
  -7.44754822e-01 -8.26438539e-01 -9.84525244e-02 -6.63478286e-01
   1.12663592e+00 -1.07993151e+00 -1.14746865e+00 -4.37820045e-01
  -4.98032451e-01  1.92953205e+00  9.49420807e-01  8.75512414e-02
  -1.22543552e+00  8.44362976e-01 -1.00021535e+00 -1.54477110e+00
   1.18802979e+00  3.16942612e-01  9.20858824e-01  3.18727653e-01
   8.56830612e-01 -6.51025593e-01 -1.03424284e+00  6.81594518e-01
  -8.03409664e-01 -6.89549778e-01 -4.55532504e-01  1.74791590e-02
  -3.53993911e-01 -1.37495129e+00 -6.43618403e-01 -2.22340315e+00
   6.25231451e-01 -1.60205766e+00 -1.10438334e+00  5.21650793e-02
  -7.39562996e-01  1.54301460e+00 -1.29285691e+00  2.67050869e-01
  -3.92828182e-02 -1.16809350e+00  5.23276661e-01 -1.71546331e-01
   7.71790551e-01  8.23504154e-01  2.16323595e+00  1.33652795e+00
  -3.69181838e-01 -2.39379178e-01  1.09965960e+00  6.55263731e-01
   6.40131526e-01 -1.61695604e+00 -2.43261244e-02 -7.38030909e-01
   2.79924599e-01 -9.81503896e-02  9.10178908e-01  3.17218215e-01
   7.86327962e-01 -4.66419097e-01 -9.44446256e-01 -4.10049693e-01
  -1.70204139e-02  3.79151736e-01  2.25930895e+00 -4.22571517e-02
  -9.55945000e-01 -3.45981776e-01 -4.63595975e-01  4.81481474e-01
  -1.54079701e+00  6.32619942e-02  1.56506538e-01  2.32181036e-01
  -5.97316069e-01 -2.37921730e-01 -1.42406091e+00 -4.93319883e-01
  -5.42861476e-01  4.16050046e-01 -1.15618243e+00  7.81198102e-01
   1.49448454e+00 -2.06998503e+00  4.26258731e-01  6.76908035e-01
  -6.37437026e-01 -3.97271814e-01 -1.32880578e-01 -2.97790879e-01
  -3.09012969e-01 -1.67600381e+00  1.15233156e+00  1.07961859e+00
  -8.13364259e-01 -1.46642433e+00  5.21064876e-01 -5.75787970e-01
   1.41953163e-01 -3.19328417e-01  6.91538751e-01  6.94749144e-01]
 [-7.25597378e-01 -1.38336396e+00 -1.58293840e+00  6.10379379e-01
  -1.18885926e+00 -5.06816354e-01 -5.96314038e-01 -5.25672963e-02
  -1.93627981e+00  1.88778597e-01  5.23891024e-01  8.84220870e-02
  -3.10886172e-01  9.74001663e-02  3.99046346e-01 -2.77259276e+00
   1.95591231e+00  3.90093323e-01 -6.52408582e-01 -3.90953375e-01
   4.93741777e-01 -1.16103939e-01 -2.03068447e+00  2.06449286e+00
  -1.10540657e-01  1.02017271e+00 -6.92049848e-01  1.53637705e+00
   2.86343689e-01  6.08843834e-01 -1.04525337e+00  1.21114529e+00
   6.89818165e-01  1.30184623e+00 -6.28087560e-01 -4.81027118e-01
   2.30391670e+00 -1.06001582e+00 -1.35949701e-01  1.13689136e+00
   9.77249677e-02  5.82953680e-01 -3.99449029e-01  3.70055888e-01
  -1.30652685e+00  1.65813068e+00 -1.18164045e-01 -6.80178204e-01
   6.66383082e-01 -4.60719787e-01 -1.33425847e+00 -1.34671751e+00
   6.93773153e-01 -1.59573438e-01 -1.33701560e-01  1.07774381e+00
  -1.12682581e+00 -7.30677753e-01 -3.84879809e-01  9.43515893e-02
  -4.21714513e-02 -2.86887192e-01 -6.16264021e-02 -1.07305276e-01
  -7.19604389e-01 -8.12992989e-01  2.74516358e-01 -8.90915083e-01
  -1.15735526e+00 -3.12292251e-01 -1.57667016e-01  2.25672350e+00
  -7.04700276e-01  9.43260725e-01  7.47188334e-01 -1.18894496e+00
   7.73252977e-01 -1.18388064e+00 -2.65917224e+00  6.06319524e-01
  -1.75589058e+00  4.50934462e-01 -6.84010898e-01  1.65955080e+00
   1.06850940e+00 -4.53385804e-01 -6.87837611e-01 -1.21407740e+00
  -4.40922632e-01 -2.80355495e-01 -3.64693544e-01  1.56703855e-01
   5.78521498e-01  3.49654457e-01 -7.64143924e-01 -1.43779147e+00
   1.36453185e+00 -6.89449185e-01 -6.52293600e-01 -5.21189312e-01
  -1.84306955e+00 -4.77974004e-01 -4.79655814e-01  6.20358298e-01
   6.98457149e-01  3.77088909e-03  9.31848374e-01  3.39964984e-01
  -1.56821116e-02  1.60928168e-01 -1.90653494e-01 -3.94849514e-01
  -2.67733537e-01 -1.12801133e+00  2.80441705e-01 -9.93123611e-01
   8.41631264e-01 -2.49458580e-01  4.94949817e-02  4.93836776e-01
   6.43314465e-01 -1.57062341e+00 -2.06903676e-01  8.80178912e-01
  -1.69810582e+00  3.87280475e-01 -2.25556423e+00 -1.02250684e+00]
 [ 3.86305518e-02 -1.65671510e+00 -9.85510738e-01 -1.47183501e+00
   1.64813493e+00  1.64227755e-01  5.67290278e-01 -2.22675101e-01
  -3.53431749e-01 -1.61647419e+00 -2.91837363e-01 -7.61492212e-01
   8.57923924e-01  1.14110187e+00  1.46657872e+00  8.52551939e-01
  -5.98653937e-01 -1.11589699e+00  7.66663182e-01  3.56292817e-01
  -1.76853845e+00  3.55481793e-01  8.14519822e-01  5.89255892e-02
  -1.85053671e-01 -8.07648488e-01 -1.44653470e+00  8.00297949e-01
  -3.09114445e-01 -2.33466662e-01  1.73272119e+00  6.84501107e-01
   3.70825001e-01  1.42061805e-01  1.51999486e+00  1.71958931e+00
   9.29505111e-01  5.82224591e-01 -2.09460307e+00  1.23721914e-01
  -1.30106954e-01  9.39532294e-02  9.43046087e-01 -2.73967717e+00
  -5.69312053e-01  2.69904355e-01 -4.66845546e-01 -1.41690611e+00
   8.68963487e-01  2.76871906e-01 -9.71104570e-01  3.14817205e-01
   8.21585712e-01  5.29264630e-03  8.00564803e-01  7.82601752e-02
  -3.95228983e-01 -1.15942052e+00 -8.59307670e-02  1.94292938e-01
   8.75832762e-01 -1.15107468e-01  4.57415606e-01 -9.64612014e-01
  -7.82629156e-01 -1.10389299e-01 -1.05462846e+00  8.20247837e-01
   4.63130329e-01  2.79095764e-01  3.38904125e-01  2.02104356e+00
  -4.68864188e-01 -2.20144129e+00  1.99300197e-01 -5.06035410e-02
  -5.17519043e-01 -9.78829859e-01 -4.39189522e-01  1.81338429e-01
  -5.02816701e-01  2.41245368e+00 -9.60504382e-01 -7.93117363e-01
  -2.28862004e+00  2.51484415e-01 -2.01640663e+00 -5.39454633e-01
  -2.75670535e-01 -7.09727966e-01  1.73887268e+00  9.94394391e-01
   1.31913688e+00 -8.82418819e-01  1.12859406e+00  4.96000946e-01
   7.71405949e-01  1.02943883e+00 -9.08763246e-01 -4.24317621e-01
   8.62596011e-01 -2.65561909e+00  1.51332808e+00  5.53132064e-01
  -4.57039607e-02  2.20507656e-01 -1.02993528e+00 -3.49943365e-01
   1.10028434e+00  1.29802197e+00  2.69622405e+00 -7.39246663e-02
  -6.58552967e-01 -5.14233966e-01 -1.01804188e+00 -7.78547559e-02
   3.82732430e-01 -3.42422805e-02  1.09634685e+00 -2.34215801e-01
  -3.47450652e-01 -5.81268477e-01 -1.63263453e+00 -1.56776772e+00
  -1.17915793e+00  1.30142807e+00  8.95260273e-01  1.37496407e+00]
 [-1.33221165e+00 -1.96862469e+00 -6.60056320e-01  1.75818953e-01
   4.98690275e-01  1.04797216e+00  2.84279671e-01  1.74266878e+00
  -2.22605681e-01 -9.13079218e-01 -1.68121822e+00 -8.88971358e-01
   2.42117961e-01 -8.88720257e-01  9.36742464e-01  1.41232771e+00
  -2.36958691e+00  8.64052300e-01 -2.23960406e+00  4.01499055e-01
   1.22487056e+00  6.48561063e-02 -1.27968917e+00 -5.85431204e-01
  -2.61645446e-01 -1.82244784e-01 -2.02896841e-01 -1.09882779e-01
   2.13480049e-01 -1.20857365e+00 -2.42019830e-01  1.51826117e+00
  -3.84645423e-01 -4.43836093e-01  1.07819730e+00 -2.55918467e+00
   1.18137860e+00 -6.31903758e-01  1.63928572e-01  9.63213559e-02
   9.42468119e-01 -2.67594746e-01 -6.78025782e-01  1.29784579e+00
  -2.36417382e+00  2.03341817e-02 -1.34792542e+00 -7.61573388e-01
   2.01125668e+00 -4.45954265e-02  1.95069697e-01 -1.78156286e+00
  -7.29044659e-01  1.96557401e-01  3.54757693e-01  6.16886554e-01
   8.62789892e-03  5.27004208e-01  4.53781913e-01 -1.82974041e+00
   3.70057219e-02  7.67902408e-01  5.89879821e-01 -3.63858810e-01
  -8.05626508e-01 -1.11831192e+00 -1.31054012e-01  1.13307988e+00
  -1.95180410e+00 -6.59891730e-01 -1.13980246e+00  7.84957521e-01
  -5.54309627e-01 -4.70637658e-01 -2.16949570e-01  4.45393251e-01
  -3.92388998e-01 -3.04614305e+00  5.43311891e-01  4.39042958e-01
  -2.19541028e-01 -1.08403662e+00  3.51780111e-01  3.79235534e-01
  -4.70032883e-01 -2.16731471e-01 -9.30156503e-01 -1.78589092e-01
  -1.55042935e+00  4.17318821e-01 -9.44368491e-01  2.38103148e-01
  -1.40596292e+00 -5.90057646e-01 -1.10489405e-01 -1.66069981e+00
   1.15147873e-01 -3.79147563e-01 -1.74235620e+00 -1.30324275e+00
   6.05120084e-01  8.95555986e-01 -1.31908640e-01  4.04761812e-01
   2.23843563e-01  3.29622982e-01  1.28598401e+00 -1.50699840e+00
   6.76460732e-01 -3.82008956e-01 -2.24258934e-01 -3.02249730e-01
  -3.75147117e-01 -1.22619619e+00  1.83339199e-01  1.67094303e+00
  -5.61330204e-02 -1.38504274e-03 -6.87299037e-01 -1.17474546e-01
   4.66166426e-01 -3.70242441e-01 -4.53804041e-01  4.03264540e-01
  -9.18004770e-01  2.52496627e-01  8.20321797e-01  1.35994854e+00]]
['Holberton', 'school', 'is', 'the', 'best!']
ubuntu-xenial:0x0B-face_verification$
```

## Task17 - Embedding
Update the class FaceVerification:<br>
<br>
public instance method `def embedding(self, images):` that calculates the face embedding of images
* images is a numpy.ndarray of shape (i, n, n, 3) containing the aligned images
* * i is the number of images
* * n is the size of the aligned images
Returns: a numpy.ndarray of shape (i, e) containing the embeddings where e is the dimensionality of the embeddings

```
ubuntu-xenial:0x0B-face_verification$.cat 7-main
#!/usr/bin/env python3

import numpy as np
from verification import FaceVerification
from utils import load_images

images, _ = load_images('HBTNaligned', as_array=True)

np.random.seed(0)
database = np.random.randn(5, 128)
identities = ['Holberton', 'school', 'is', 'the', 'best!']
fv = FaceVerification('models/trained_fv.h5', database, identities)
embs = fv.embedding(images)
print(embs.shape)

ubuntu-xenial:0x0B-face_verification$
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
(385, 128)
ubuntu-xenial:0x0B-face_verification$
```

## Task18 - Verify
Update the class FaceVerification:<br>
<br>
public instance method `def verify(self, image, tau=0.5):`
* image is a numpy.ndarray of shape (n, n, 3) containing the aligned image of the face to be verify
* * n is the shape of the aligned image
* tau is the maximum euclidean distance used for verification<br>
Returns: (identity, distance), or (None, None) on failure
* identity is a string containing the identity of the verified face
* distance is the euclidean distance between the verified face embedding and the identified database embedding

```
$ubuntu-xenial:0x0B-face_verification$cat 18-main
#!/usr/bin/env python3

from matplotlib.pyplot import imread, imsave
import numpy as np
import tensorflow as tf
from utils import load_images
from verification import FaceVerification

database_files = ['HBTNaligned/HeimerRojas.jpg', 'HBTNaligned/MariaCoyUlloa.jpg', 'HBTNaligned/MiaMorton.jpg', 'HBTNaligned/RodrigoCruz.jpg', 'HBTNaligned/XimenaCarolinaAndradeVargas.jpg']
database_imgs = np.zeros((5, 96, 96, 3))
for i, f in enumerate(database_files):
    database_imgs[i] = imread(f)

database_imgs = database_imgs.astype('float32') / 255

with tf.keras.utils.CustomObjectScope({'tf': tf}):
    base_model = tf.keras.models.load_model('models/face_verification.h5')
    database_embs = base_model.predict(database_imgs)

test_img_positive = imread('HBTNaligned/HeimerRojas0.jpg').astype('float32') / 255
test_img_negative = imread('HBTNaligned/KirenSrinivasan.jpg').astype('float32') / 255

identities = ['HeimerRojas', 'MariaCoyUlloa', 'MiaMorton', 'RodrigoCruz', 'XimenaCarolinaAndradeVargas']
fv = FaceVerification('models/face_verification.h5', database_embs, identities)
print(fv.verify(test_img_positive, tau=0.46))
print(fv.verify(test_img_negative, tau=0.46))
ubuntu-xenial:0x0B-face_verification$./18-main
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
('HeimerRojas', 0.42472672)
(None, None)
ubuntu-xenial:0x0B-face_verification$
```
