import numpy as np
import cv2
import os 
from scipy.linalg import eigh
from numpy.linalg import norm

class FaceRecognition:
    def __init__(self,tab_widget,file_path):
        self.ui = tab_widget
        self.test_data_path = file_path
        self.eigen_values, self.eigen_vectors,self.eigen_faces = None, None, None



    def train_data(self):
        folder_path = r"Images\Train image faces"
        self.persons_name = []
        self.X_train = [] 
        for folder in os.listdir(folder_path):
            person_name = folder
            directory_path = os.path.join(folder_path,person_name)
            for image in os.listdir(directory_path):
                image_path = os.path.join(directory_path,image)     
                img_data = cv2.imread(image_path,0)
                img_data = cv2.resize(img_data,(50,50))
                img_data = np.array(img_data)
                img_data = np.reshape(img_data,[1,img_data.shape[0]*img_data.shape[1]])
                self.X_train.append(img_data)
                self.persons_name.append(person_name)    
        self.X_train = np.array(self.X_train)
        self.X_train = np.reshape(self.X_train,(self.X_train.shape[0],-1))        


    def Normalize_data(self):
        mean_data = np.mean(self.X_train,axis=0)
        return mean_data        
    


    
    def get_eigenVectors(self):
        mean = self.Normalize_data()
        normalized_data = self.X_train - mean
        ## get covariance matrix
        cov_mat = np.cov(normalized_data.T,dtype=np.float32)
        
        ## get eigen vectors and values
        eigen_values, eigen_vectors = eigh(cov_mat)
        
        ## sorted eigen values and eigen vectors descending
        
        sorted_eigen_values_indices = (-eigen_values).argsort()
        sorted_eigen_vectors = eigen_vectors[:,sorted_eigen_values_indices]
        
        return eigen_values, sorted_eigen_vectors, sorted_eigen_values_indices    

    
    def get_the_best_numberOfEigenvectors(self,sorted_eigen_values_indices):
        ##  stop at pov > 0.98
        max_pov = 0.98
        K = 0
        sum_eigen_values = 0
        total_eigen_values = np.sum(self.eigen_values)
        for i,K in enumerate(sorted_eigen_values_indices):
            sum_eigen_values += self.eigen_values[K]
            POV = sum_eigen_values/total_eigen_values
            # print(f"pov:{POV}")
            POV = np.round(POV,2)
            if POV > max_pov:
                no_of_eigen_vectors= i+1
                break
            else:
                no_of_eigen_vectors = len(self.eigen_values)
                
        print(f"no of features:{no_of_eigen_vectors}")
        return no_of_eigen_vectors      

    def ReconstructImage(self,x_test_data,reconstruct=False):
        x_test_data = x_test_data - self.mean_data
        x_test_data = np.matmul(x_test_data,self.eigen_faces)
        if reconstruct:
            x_test_data = np.matmul(x_test_data,self.eigen_faces.T)
            x_test_data = x_test_data + self.mean_data
            x_test_data = np.reshape(x_test_data,[50,50])
            x_test_data =x_test_data.astype("uint8")    
        return x_test_data  
    

    def cross_correlation(self,trained_data,tested_data):
        nor = np.sqrt(np.sum(trained_data**2)) * np.sqrt(np.sum(tested_data**2))
        score = np.sum(trained_data*tested_data)/ nor
        return score


    def sum_square_difference(self,x,y):
        nor = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
        score = np.sum((x-y)**2)/ nor
        return score

    def cosine_similerity(self,x,y):
        similerity_index = np.dot(x,y)/(norm(x)*norm(y))
        return similerity_index
    
    def eculidean_distance(self,point1,point2):
        return np.sqrt(np.sum((point1-point2)**2))
    

    def pca(self):
        self.train_data()
        self.mean_data = self.Normalize_data()
        self.eigen_values, self.eigen_vectors,sorted_eigen_values_indices = self.get_eigenVectors()
        no_of_eigen_vectors = self.get_the_best_numberOfEigenvectors(sorted_eigen_values_indices)
        self.eigen_faces = self.eigen_vectors[:,:(no_of_eigen_vectors)]
    
    


    def run_model(self):## run model
        self.pca()

    
    def test_data(self):
        self.run_model()
        ##  matching image
        test_data = cv2.imread(self.test_data_path ,0)
        test_data = cv2.resize(test_data,(50,50))
        test_data = np.array(test_data)
        test_data = np.reshape(test_data,[1,test_data.shape[0]*test_data.shape[1]])[0]
        tested_image_projected = self.ReconstructImage(test_data)


        tested_image = self.ReconstructImage(test_data,True)
        self.ui.display_image(self.ui.graphicsLayout_AfterFaceRecognition,tested_image) ## display_image
        scores_list = []

        for image in self.X_train:
            trained_image_projected = self.ReconstructImage(image)
            score = self.eculidean_distance(tested_image_projected,trained_image_projected)
            scores_list.append(score)

        predicted_person_idx = np.argmin(scores_list)    
        predicted_person = self.persons_name[predicted_person_idx]
        self.ui.label_personName.setText(str(predicted_person))
        print(f"predicted_person:{predicted_person}")