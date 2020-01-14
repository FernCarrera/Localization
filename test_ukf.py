import ukf 
import unittest
import numpy as np

class TestUkf(unittest.TestCase):

    def test_dist(self):
        pos = [0,0,0]; pt1 = [5,5]
        sigma_range = 0; sigma_bearing = 0
        
        dist_formula = np.sqrt( (pt1[0]-pos[0])**2 + (pt1[1]-pos[1])**2 )
        result = ukf.dist(pt1,pos,sigma_range,sigma_bearing)
        
        self.assertEqual(result[0],dist_formula)
        
    def test_closest_point(self):
        x = np.arange(0,10,1)
        y = np.arange(0,10,1)
        path_list = [[x,y] for x,y in zip(x,y)]
        bumper = 3
        pos = [4,8,0]
        path_dict = dict((x,y) for x,y in zip(x,y) )
        c_point = ukf.closest_point(pos,path_dict,bumper)
        
        # return point in line
        self.assertIn(c_point, path_list)  
        # check for points outside the path

        # check for bumper making to check outside line





if __name__ =='__main__':
    unittest.main()
