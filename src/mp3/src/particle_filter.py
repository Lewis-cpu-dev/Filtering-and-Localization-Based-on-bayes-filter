import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

import random

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # (Default) The whole map
            # x = np.random.uniform(0, world.width)
            # y = np.random.uniform(0, world.height)


            ## first quadrant
            x = np.random.uniform(world.width / 2, world.width)
            y = np.random.uniform(world.height / 2, world.height)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

        ###############

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 9000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####

        if readings_robot is None:
        # No robot reading → uniform weights
            for p in self.particles:
                p.weight = 1.0 / self.num_particles
            return

        sigma = 3000.0 
        total = 0 
        for p in self.particles:
            readings_particle = p.read_sensor()
            # error = np.array(readings_robot) - np.array(readings_particle)
            # prob = np.exp(-0.5 * (error**2 / sigma))
            # # Multiply likelihoods across all directions
            # weight = np.sum(prob)
            # p.weight = weight
            # for i in range(len(readings_particle)):
            p.weight = self.weight_gaussian_kernel(readings_robot,readings_particle,sigma)
                 
        # Normalize weights
            total +=p.weight
        if total > 0:
            for p in self.particles:
                p.weight /= total
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

            ###############
            # pass

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        weights = [p.weight for p in self.particles]
        indices = list(range(len(self.particles)))
        cumsum = np.cumsum(weights)
        step = 1.0 / self.num_particles
        r = random.uniform(0, step)
        i = 0
        for m in range(self.num_particles):
            u = r + m * step
            while u > cumsum[i]:
                i += 1
            new_p = Particle(
                x=self.particles[i].x,
                y=self.particles[i].y,
                heading=self.particles[i].heading,
                maze=self.world,
                sensor_limit=self.sensor_limit,
                weight=1.0
            )
            particles_new.append(new_p)

        self.particles = particles_new



    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO #####
        if not self.control:
            return

        control = self.control.pop(0)
        vr = control[0]
        delta = control[1]
        dt = 0.01

        for p in self.particles:
            solver = ode(vehicle_dynamics).set_integrator('dopri5')
            solver.set_initial_value([p.x, p.y, p.heading], 0)
            solver.set_f_params(vr, delta)
            sol = solver.integrate(dt)
            p.x = sol[0]
            p.y = sol[1]
            p.heading = sol[2]
            p.fix_invalid_particles()

        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0
        pos_errors = []
        heading_errors = []
        self.world.show_maze()
        while not rospy.is_shutdown():
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            
            ###############         
            rospy.sleep(0.1)
            self.particleMotionModel()
            readings_robot = self.bob.read_sensor()
            self.updateWeight(readings_robot)
            self.resampleParticle()

            # Visualize
            self.world.clear_objects()
            
            self.world.show_particles(self.particles, show_frequency=5)
            estimate = self.world.show_estimated_location(self.particles)
            self.world.show_robot(self.bob)

            # Log position and heading errors
            if estimate:
                actual_x, actual_y, actual_heading = self.bob.x, self.bob.y, self.bob.heading
                est_x, est_y, est_heading = estimate

                pos_error = np.sqrt((actual_x - est_x)**2 + (actual_y - est_y)**2)
                heading_error = abs((actual_heading - np.deg2rad(est_heading)) % (2*np.pi))
                if heading_error > np.pi:
                    heading_error = 2*np.pi - heading_error

                pos_errors.append(pos_error)
                heading_errors.append(heading_error)

            count += 1
            if count >= 1000:  # Optional stop condition
                break

        # Save or return errors for plotting later
        np.save("position_errors.npy", np.array(pos_errors))
        np.save("heading_errors.npy", np.array(heading_errors))
