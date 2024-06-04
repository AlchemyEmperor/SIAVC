import numpy as np



def OTSU_threshold(H_list):
    
    losses_array = np.array(H_list)

    
    best_threshold = 0
    best_variance = 0



 
    for threshold in range(256):
       
        class1 = losses_array[losses_array <= threshold]
        class2 = losses_array[losses_array > threshold]

      
        w1 = len(class1) / len(losses_array)
        w2 = len(class2) / len(losses_array)

       
        mean1 = np.mean(class1)
        mean2 = np.mean(class2)

     
        between_class_variance = w1 * w2 * ((mean1 - mean2) ** 2)

      
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = threshold

    return best_threshold
