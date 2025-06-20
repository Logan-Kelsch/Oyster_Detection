# anno_img.py
import cv2
# … load your YOLO model here …

def run(input_path: str, output_path: str) -> str:
    img = cv2.imread(input_path)
    # … do your detection & draw boxes …
    cv2.imwrite(output_path, img)
    return output_path

if __name__ == "__main__":
    import sys
    in_p, out_p = sys.argv[1], sys.argv[2]
    run(in_p, out_p)
