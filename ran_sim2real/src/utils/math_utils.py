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

