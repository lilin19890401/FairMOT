# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    在目标跟踪中，需要估计track的以下两个状态：
    均值(Mean)：表示目标的位置信息，由bbox的中心坐标 (cx, cy)，宽高比a，高h，以及各自的速度变化值组成，由8维向量表示为 x = [cx, cy, a, h, vx, vy, va, vh]，各个速度值初始化为0。
    协方差(Covariance )：表示目标位置信息的不确定性，由8x8的对角矩阵表示，矩阵中数字越大则表明不确定性越大，可以以任意值初始化

    卡尔曼滤波分为两个阶段：(1) 预测track在下一时刻的位置，(2) 基于detection来更新预测的位置
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)           # 这里是指状态转移矩阵，F表示
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)               # 这里是指测量矩阵，它将track的均值向量x'映射到检测空间，H表示

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement                                  # 测量得到的均值（x,y,a,h）
        mean_vel = np.zeros_like(mean_pos)                      # vx,vy,xa,vh 这四个值初始时赋0
        mean = np.r_[mean_pos, mean_vel]                        # 形成均值向量

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]    # 初始化标准差
        covariance = np.diag(np.square(std))                    # 初始协方差矩阵
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        卡尔曼滤波预测步骤:
            基于tracker在t-1时刻的状态来预测其在t时刻的状态
            x' = Fx                 (1)
            P' = FPF(T) + Q         (2)
            在公式1中，x为tracker在t-1时刻的均值，F称为状态转移矩阵，该公式预测t时刻的x'：
            |cx‘|            | 1  0  0  0  dt  0  0  0 |     |cx|
            |cy’|            | 0  1  0  0  0   dt 0  0 |     |cy|
            |a‘ |            | 0  0  1  0  0   0  dt 0 |     |a |
            |h’ |     =      | 0  0  0  1  0   0  0  dt|  *  |h |
            |vx‘|            | 0  0  0  0  1   0  0  0 |     |vx|
            |vy’|            | 0  0  0  0  0   1  0  0 |     |vy|
            |va‘|            | 0  0  0  0  0   0  1  0 |     |va|
            |vh’|            | 0  0  0  0  0   0  0  1 |     |vh|
            t时刻                                             t-1时刻
            x'                          F                    x

            矩阵F中的dt是当前帧和前一帧之间的差，将等号右边的矩阵乘法展开，可以得到cx'=cx+dt*vx，cy'=cy+dt*vy...，所以这里的卡尔曼滤波是一个匀速模型（Constant Velocity Model）
            在公式2中，P为tracker在t-1时刻的协方差，Q为系统的噪声矩阵，代表整个系统的可靠程度，一般初始化为很小的值，该公式预测t时刻的P'
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))                    # 初始化系统噪声矩阵Q

        #mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)                                     # x' = Fx, F为状态转移矩阵 // 得到t时刻的均值
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov         # P' = FPF(T) + Q       // 得到t时刻的协方差

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))                    # 初始化传感器噪声矩阵R

        mean = np.dot(self._update_mat, mean)       				# 将均值向量映射到检测空间,即公式(3)中的 Hx'
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))      # 将协方差矩阵映射到检测空间，即HP'H(T)，公式(4)中的 HP'H(T)
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        卡尔曼滤波预测步骤:
            基于tracker在t-1时刻的状态来预测其在t时刻的状态，这里输入量都是多维的，一次性送入多个目标
            x' = Fx                 (1)
            P' = FPF(T) + Q         (2)
            在公式1中，x为tracker在t-1时刻的均值，F称为状态转移矩阵，该公式预测t时刻的x'：
            |cx‘|            | 1  0  0  0  dt  0  0  0 |     |cx|
            |cy’|            | 0  1  0  0  0   dt 0  0 |     |cy|
            |a‘ |            | 0  0  1  0  0   0  dt 0 |     |a |
            |h’ |     =      | 0  0  0  1  0   0  0  dt|  *  |h |
            |vx‘|            | 0  0  0  0  1   0  0  0 |     |vx|
            |vy’|            | 0  0  0  0  0   1  0  0 |     |vy|
            |va‘|            | 0  0  0  0  0   0  1  0 |     |va|
            |vh’|            | 0  0  0  0  0   0  0  1 |     |vh|
            t时刻                                             t-1时刻
            x'                          F                    x

            矩阵F中的dt是当前帧和前一帧之间的差，将等号右边的矩阵乘法展开，可以得到cx'=cx+dt*vx，cy'=cy+dt*vy...，所以这里的卡尔曼滤波是一个匀速模型（Constant Velocity Model）
            在公式2中，P为tracker在t-1时刻的协方差，Q为系统的噪声矩阵，代表整个系统的可靠程度，一般初始化为很小的值，该公式预测t时刻的P'
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # 初始化系统噪声矩阵Q
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)         					# x' = Fx, F为状态转移矩阵 // 得到t时刻的均值
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov      	# P' = FPF(T) + Q       // 得到t时刻的协方差

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        卡尔曼滤波更新操作:基于t时刻检测到的detection，校正与其关联的track的状态，得到一个更精确的结果
        y = z - Hx'       (3)
        S = HP'H(T) + R   (4)
        K = P'H(T)S(-1)   (5)
        x = x' + Ky       (6)
        P = (I - KH)P'    (7)
        在公式3中，z为detection的均值向量，不包含速度变化值，即z=[cx, cy, r, h]，H称为测量矩阵，它将tracker的均值向量x'映射到检测空间，该公式计算detection和track的均值误差;
        在公式4中，R为检测器的噪声矩阵，它是一个4x4的对角矩阵，对角线上的值分别为中心点两个坐标以及宽高的噪声，以任意值初始化，一般设置宽高的噪声大于中心点的噪声，
        该公式先将协方差矩阵P'映射到检测空间，然后再加上噪声矩阵R;
        公式5计算卡尔曼增益K，卡尔曼增益用于估计误差的重要程度;
        公式6和公式7得到更新后的均值向量x和协方差矩阵P
        """

        # 将mean和covariance映射到检测空间，得到Hx'和S
        projected_mean, projected_cov = self.project(mean, covariance)      # 公式(3),(4)

        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean       					# 公式(3) y = z - Hx‘

        new_mean = mean + np.dot(innovation, kalman_gain.T)     			# 公式(6) x = x' + Ky
        new_covariance = covariance - np.linalg.multi_dot((     			# 公式(7) P = (I - KH)P'
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        """
        Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。它要求矩阵的所有特征值必须大于零，故分解的下三角的对角元也是大于零的。
        Cholesky分解法又称平方根法，是当A为实对称正定矩阵时，LU三角分解法的变形。与一般的矩阵分解求解方程的方法比较，Cholesky分解效率很高。
        可记作A = L * L.H。其中L是下三角矩阵。L.H是L的共轭转置矩阵。
        当线性方程组 Ax=b可用Cholesky分解法求解时，Cholesky分解法的求解效率大约是LU分解法的2倍
        
        scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, debug=None, check_finite=True)
        Solve the equation a x = b for x, assuming a is a triangular matrix.
        Parameters
                a: (M, M) array_like
                    A triangular matrix
                b: (M,) or (M, N) array_like
                    Right-hand side matrix in a x = b

                lower: bool, optional
                    Use only data contained in the lower triangle of a. Default is to use upper triangle.
                trans: {0, 1, 2, ‘N’, ‘T’, ‘C’}, optional
                    Type of system to solve:
                        trans           system
                        0 or ‘N’        a x = b
                        1 or ‘T’        a^T x = b     
                        2 or ‘C’        a^H x = b
                unit_diagonal: bool, optional
                    If True, diagonal elements of a are assumed to be 1 and will not be referenced.

                overwrite_b: bool, optional
                    Allow overwriting data in b (may enhance performance)

                check_finite: bool, optional
                    Whether to check that the input matrices contain only finite numbers. Disabling may give a performance gain, 
                    but may result in problems (crashes, non-termination) if the inputs do contain infinities or NaNs.        

        Returns x: (M,) or (M, N) ndarray
                    Solution to the system a x = b. Shape of return matches b.

        Raises  LinAlgError          
                    If a is singular

        Examples
            Solve the lower triangular system a x = b, where:

                     [3  0  0  0]       [4]
                a =  [2  1  0  0]   b = [2]
                     [1  0  1  0]       [4]
                     [1  1  1  1]       [2]

                from scipy.linalg import solve_triangular
                a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
                b = np.array([4, 2, 4, 2])
                x = solve_triangular(a, b, lower=True)
                x
                array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
                a.dot(x)  # Check the result
                array([ 4.,  2.,  4.,  2.])
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')