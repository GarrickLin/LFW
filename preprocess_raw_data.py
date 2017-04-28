import cPickle as pickle
import numpy as np
import cv2


def draw_shape(img, shape):
    for pt in shape:
        cv2.circle(img, tuple(map(int, pt)), 2, (255,0,0))
    return img


def format_embedding(emstr):
    emstr = emstr.strip()[1:-1].split(',')
    emfloat = map(float, emstr)
    return np.array(emfloat)


def preprocess(filename):
    lfw_dict = dict()
    ce = np.array([125., 125.])
    with open(filename) as f:
        cnt = 0
        cnt1 = 0
        while 1:
            try:
                name = next(f).strip().split('.')[0]
                print "name", name
                num = int(next(f))      
                if num == 1:
                    pts = next(f)
                    embedding = next(f)
                    embedding = format_embedding(embedding)
                    lfw_dict[name] = embedding
                elif num > 1:
                    min_dist = 500
                    draw = np.ones((250,250,3), dtype=np.uint8) * 255                    
                    for i in range(num):                        
                        pts = next(f)
                        pts = pts.rstrip(",\n").split(',')
                        pts = np.array(pts, dtype=float)
                        pts = np.reshape(pts, (21, 2))
                        center = np.mean(pts, axis=0)                        
                        draw = draw_shape(draw, [center])     
                        dist = np.linalg.norm(center-ce)
                        embedding = next(f)   
                        if dist < min_dist:
                            embedding = format_embedding(embedding)
                            lfw_dict[name] = embedding                            
                            min_dist = dist           
                            print "update embedding"
                            min_ce = center
                    cv2.circle(draw, (int(min_ce[0]), int(min_ce[1])), 4, (0,0,255))
                    print num
                    cv2.imshow("draw", draw)
                    cv2.waitKey(1)                    

            except StopIteration:
                print "there is no more data"
                break
            
    pickle.dump(lfw_dict, 
                open("data/lfw_embedding.pkl", "wb"), 
                protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    preprocess("data/LFW.txt")
                
                