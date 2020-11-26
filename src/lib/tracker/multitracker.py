from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]    # 检测网络的检测结果
            hm = output['hm'].sigmoid_()        # 检测网络输出的热力图
            wh = output['wh']                   # 检测网络输出的目标宽高
            id_feature = output['id']           # 检测网络输出的Re-ID特征
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None    # 检测网络输出的目标中心offset
            # 检测的det res(bb, score, clses, ID)以及特征得分图的排序的有效index
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            # 根据 index 选取 有效的Re-ID特征
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            # 去除那些维度大小为1的维度
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        # 对检测结果做后处理
        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        # 检测置信度阈值过滤，得到有效的目标和对应的Re-ID特征
        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections  对每个检测目标转化为跟踪对象，并绑定检测结果等属性'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding
                    1. 将[activated_stracks lost_stracks]融合成strack_pool
                    2. detections和strack_pool根据feats计算外观cost矩阵，就是用feat计算cosine距离
                    3. 利用卡尔曼算法预测strack_pool的新的mean，covariance、
                    4. 计算strack_pool和detection的距离cost矩阵，并将大于距离阈值的外观cost矩阵赋值为inf
                    5. 利用匈牙利算法进行匹配（这里没有采用Munkres,而是利用另一种高效最优任务分配方法：LAPJV）
                        a. 能匹配成功：
                            strack_pool中的track_state==tracked，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入activated_stracks
                            strack_pool中的track_state!=tracked，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入refind_stracks
                        b. 未成功匹配：
                            得到新的detections，r_tracked_stracks
        '''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)                               # 卡尔曼预测
        dists = matching.embedding_distance(strack_pool, detections)    # 计算新检测出来的目标detections和strack_pool之间的cosine距离
        #dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)    # 利用卡尔曼计算strack_pool和detection的距离cost，并将大于距离阈值的外观cost矩阵赋值为inf（距离约束）
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)       # LAPJV匹配 // 将跟踪框和检测框进行匹配 // matches是匹配对索引，u_track是未匹配的tracker的索引，u_detection是未匹配的检测目标索引

        for itracked, idet in matches:                                  # matches:63*2 , 63:匹配成对个数，2：第一列为tracked_tracker索引，第二列为detection的索引
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)           # 匹配的tracker和detection，更新特征和卡尔曼状态
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)     # 如果是在lost中的，就重新激活
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU
                    对余弦距离未匹配剩下的detections，r_tracked_stracks进行IOU匹配
                    1. detections和r_tracked_stracks计算IOU cost矩阵
                    2. 针对IOU cost进行匈牙利匹配（这里没有采用Munkres,而是利用另一种高效最优任务分配方法：LAPJV）
                        a. 能匹配成功：
                            r_tracked_stracks中的track_state==tracked，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入activated_stracks
                            r_tracked_stracks中的track_state!=tracked，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入refind_stracks
                        b. 未成功匹配：
                            r_tracked_stracks中的状态track_state不为lost的，改为lost
                            detections再遗留到下一步进行继续匹配
        '''
        detections = [detections[i] for i in u_detection]                                                   # u_detection是上步未匹配的detection的索引
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked] # 上步没有匹配的且是跟踪状态的tracker
        dists = matching.iou_distance(r_tracked_stracks, detections)                                        # 计算IOU cost矩阵
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)                       # 针对IOU cost进行LAPJV匹配

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)                                  # 将和r_tracked_stracks iou未匹配的剩下的tracker的状态改为lost

        ''' Deal with unconfirmed tracks, usually tracks with only one beginning frame
            上一步遗留的detection与unconfirmed_stracks进行IOU匹配
            1. 计算IOU cost矩阵
            2. 匈牙利匹配（这里没有采用Munkres,而是利用另一种高效最优任务分配方法：LAPJV）
                a. 能匹配成功：
                    更新 unconfirmed_stracks，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入activated_stracks
                b. 未成功匹配：
                    unconfirmed_stracks直接计入removed_stracks
                    不能匹配的detections，再遗留到下一步
        '''
        detections = [detections[i] for i in u_detection]                   # 将cosine/iou 未匹配的detection和unconfirmed_tracker进行匹配
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)   # 更新 unconfirmed_stracks，更新smooth_feat，卡尔曼状态更新mean，covariance（卡尔曼用），计入activated_stracks
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)                                   # unconfirmed_stracks直接计入removed_stracks

        """ Step 4: Init new stracks 
            上一步遗留的detections，初始化成新的tracker，计入activated_stracks
        """
        for inew in u_detection:                                            # 对cosine/iou/uncofirmed_tracker都未匹配的detection重新初始化成一个新的tracker
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)               # 激活track，第一帧的activated=T，其他为False
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:        # 消失 max_time_lost 帧之后,计入removed_stracks，删除
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]       # 筛出tracked状态的tracker
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)                   # 向self.tracked_stacks中添加新的detection
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)          			# 重新匹配出的trackers
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
