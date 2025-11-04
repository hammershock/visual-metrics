from metrics import DINOv2Score, VGGScore, FID1KScore, CLIPScore

from PIL import Image


if __name__ == "__main__":
    # examples
    image0 = Image.open("images/room.png")
    image1 = Image.open("images/village1.png")
    image2 = Image.open("images/village2.png")
    vgg = VGGScore()
    dinov2 = DINOv2Score()
    fid1k = FID1KScore()
    clip = CLIPScore()

    for metrics in [vgg, dinov2, fid1k, clip]:
        f1 = metrics.score(image0, image1)
        f2 = metrics.score(image0, image2)
        f3 = metrics.score(image1, image2)
        print(f1, f2, f3)
        