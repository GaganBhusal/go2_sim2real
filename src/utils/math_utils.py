import torch

def _quat_to_euler(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                        torch.sign(sinp) * torch.pi / 2,
                        torch.asin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def _calculate_projected_acceleration(base_quat):
    
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    gx = 2 * (x*z - w*y)
    gy = 2 * (y*z + w*x)
    gz = 1 - 2 * (x*x + y*y)
    return torch.stack([-gx, -gy, -gz], dim=1)

def _euler_to_quat(xyz):
    out = torch.empty(xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device)

    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0

    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out