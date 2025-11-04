from metrics import DINOv2Score, VGGScore, FID1KScore

from PIL import Image


if __name__ == "__main__":
    # examples
    image0 = Image.open("images/room.png")
    image1 = Image.open("images/village1.png")
    image2 = Image.open("images/village2.png")
    # metrics = VGGScore()
    # metrics = DINOv2Score()
    metrics = FID1KScore()


    f1 = metrics.score(image0, image1)
    f2 = metrics.score(image0, image2)
    f3 = metrics.score(image1, image2)
    print(f1, f2, f3)
    