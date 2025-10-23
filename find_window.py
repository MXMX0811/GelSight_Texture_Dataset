import cv2, numpy as np

# ===================== 内部状态（跨帧记忆） =====================
_MARKER_STATE = {
    "prev_corners": None,  # 上一帧4角（tl,tr,br,bl），float32, shape=(4,2)
    "prev_roi": None       # 上一帧ROI (x,y,w,h)
}

# ===================== 工具函数 =====================
def _order_tl_tr_br_bl(pts):
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    out = np.zeros((4,2), np.float32)
    out[0] = pts[np.argmin(s)]  # tl
    out[2] = pts[np.argmax(s)]  # br
    out[1] = pts[np.argmin(d)]  # tr
    out[3] = pts[np.argmax(d)]  # bl
    return out

def _warp_by_corners(img, corners):
    c = _order_tl_tr_br_bl(corners.astype(np.float32))
    def dist(a,b): return float(np.linalg.norm(a-b))
    W = max(1, int(max(dist(c[0],c[1]), dist(c[2],c[3]))))
    H = max(1, int(max(dist(c[0],c[3]), dist(c[1],c[2]))))
    M = cv2.getPerspectiveTransform(
        c, np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    )
    return cv2.warpPerspective(img, M, (W, H))

def _rect_from_corners(c):
    x,y,w,h = cv2.boundingRect(c.astype(np.int32))
    return np.array([x,y,w,h], dtype=np.int32)

def _iou_rect(a, b):
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter + 1e-6
    return inter/union

def _shrink_corners(corners, ratio=0.015):
    """将四个角点向多边形中心缩进 ratio（1~2% 推荐）"""
    if ratio <= 0: return corners.copy()
    ctr = corners.mean(axis=0, keepdims=True)
    return (ctr + (corners - ctr) * (1.0 - float(ratio))).astype(np.float32)

# ===================== 基础检测：白色标记 → 内侧角点 =====================
def _detect_inner_corners_from_markers(frame):
    """返回 (4,2) 内侧角点（tl,tr,br,bl）；失败返回 None"""
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 提取白色标记
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 4:
        return None

    # 取面积Top10里离中心最远的4个
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    centers = np.array([np.mean(c.reshape(-1,2), axis=0) for c in cnts], dtype=np.float32)
    img_ctr = np.array([W/2, H/2], dtype=np.float32)
    idxs = np.argsort(-np.linalg.norm(centers - img_ctr, axis=1))[:4]
    selected_cnts = [cnts[i] for i in idxs]

    # 每个标记取 minAreaRect 顶点，然后选离“标记中心均值”最近的顶点=内侧角点
    boxes   = [cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32) for c in selected_cnts]
    mcenters= np.array([b.mean(axis=0) for b in boxes], dtype=np.float32)
    win_ctr = mcenters.mean(axis=0)

    inner = []
    for b in boxes:
        inner.append(b[np.argmin(np.linalg.norm(b - win_ctr[None,:], axis=1))])
    inner = np.array(inner, dtype=np.float32)

    return _order_tl_tr_br_bl(inner)

# ===================== 稳定版 API =====================
def find_window(
    frame,
    alpha=0.8,           # EMA平滑系数（越大越稳）
    min_iou=0.25,         # 当前与上一帧ROI的最小IoU，低于则判为异常
    max_jump=40.0,        # 单帧允许的角点平均位移（像素）
    shrink_ratio=0.01,   # 结果向内收缩比例（推荐 0.01~0.02）
    enforce_aspect=None   # 如 (4,3)，需要时再加
):
    """
    返回: (roi, corners, warp)
    - corners: 稳定后的4角点（tl,tr,br,bl）
    - warp: 透视矫正图，一定非空
    """
    H, W = frame.shape[:2]
    full_corners = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)

    # 1) 基础检测
    cur = _detect_inner_corners_from_markers(frame)

    # 2) 失败 → 回退
    if cur is None:
        cur = _MARKER_STATE["prev_corners"] if _MARKER_STATE["prev_corners"] is not None else full_corners

    # 3) 若有上一帧 → 一致性判定 + 指数平滑
    prev = _MARKER_STATE["prev_corners"]
    if prev is not None:
        # 几何一致性：IoU + 平均位移
        cur_roi  = _rect_from_corners(cur)
        prev_roi = _rect_from_corners(prev)
        iou = _iou_rect(cur_roi, prev_roi)
        mean_jump = float(np.linalg.norm((cur - prev), axis=1).mean())

        if iou < min_iou and mean_jump > max_jump:
            # 异常帧，直接沿用上一帧
            cur = prev.copy()
        else:
            # 指数平滑
            cur = (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)

    # 4) 向内收缩
    cur = _shrink_corners(cur, ratio=shrink_ratio)

    # 5) 可选：强制宽高比（只收不放，防止越界）
    if enforce_aspect is not None:
        aw, ah = enforce_aspect
        # 以当前四角的包围矩形为基，收缩到目标比例
        x,y,w,h = _rect_from_corners(cur)
        tw = int(round(h * aw / ah))
        th = int(round(w * ah / aw))
        if tw <= w:
            cx = x + w//2; x = int(cx - tw//2); w = tw
        else:
            cy = y + h//2; y = int(cy - th//2); h = th
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
        cur = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.float32)

    # 6) 生成输出
    roi  = tuple(_rect_from_corners(cur))
    warp = _warp_by_corners(frame, cur)

    # 7) 更新状态
    _MARKER_STATE["prev_corners"] = cur.copy()
    _MARKER_STATE["prev_roi"]     = np.array(roi, dtype=np.float32)

    return roi, cur, warp

def reset_marker_tracker():
    """需要时手动清空平滑状态（比如摄像头重启或场景切换）"""
    _MARKER_STATE["prev_corners"] = None
    _MARKER_STATE["prev_roi"] = None
