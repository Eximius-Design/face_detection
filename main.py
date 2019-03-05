import argparse
from detect_face import detect
import tensorflow as tf
import cv2
from imutils import paths

def main():
    model_path = './mtcnn.pb'
    imagePaths = list(paths.list_images('/home/ravikiranb/homedir/images/train_img/somaliya/'))
    j = 0
    graph=tf.Graph()
    with graph.as_default():
        with open(model_path, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())
            tf.import_graph_def(graph_def, name='')
    config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=4,inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    for i in imagePaths:
        j = j+1
        img = cv2.imread(i,1)
        img_h, img_w = img.shape[:2]
        img_a = img_h*img_w

        bbox, scores, landmarks = detect(img,sess, graph, min_size=40, factor=0.709, thresholds=[0.6,0.7,0.7])

        print('total box:', len(bbox))
        for box, pts in zip(bbox, landmarks):
            box = box.astype('int32')
            box_w = box[3] - box[1]
            box_h = box[2] - box[0]
            box_a = box_w*box_h

            percent = box_a*100/img_a
            if percent > 3.0:
                print('percentage of bounding box in total image : {:.2f}'.format(percent))
                img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)

                pts = pts.astype('int32')
                for i in range(5):
                    img = cv2.circle(img, (pts[i+5], pts[i]), 4, (0, 0, 255), 8)
        cv2.imwrite('./opimages/imageeee'+str(j)+'.jpg', img)
        
if __name__ == '__main__':
    main()

