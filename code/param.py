
from numpy import array, arange, zeros

param = {}

# learning 
# rl 
param['rl_lr'] = 0.0005
param['rl_gamma'] = 0.98
param['rl_lmbda'] = 0.95
param['rl_eps_clip'] = 0.1
param['rl_K_epoch'] = 3
param['rl_n_eps'] = 5000
param['rl_model_fn'] = 'rl_model.pt'
param['rl_model_save_eps'] = 10000

# pid gain learning 
param['gains_lr'] = 0.05
param['gains_n_epoch'] = 300
param['gains_batch_size'] = 250
param['gains_n_data'] = 10000
param['gains_log_interval'] = 100
param['gains_print_epoch'] = 100
param['gains_model_fn'] = 'gain_model.pt'

# system
param['system'] = 'CartPole'

# sim
param['t0'] = 0
param['tf'] = 20
param['dt'] = 0.01
param['times'] = arange(param.get('t0'), param.get('tf'), param.get('dt'))
param['nt'] = len(param.get('times'))

# pid 
param['kp'] = 0.0001
param['kv'] = 0.00
param['ki'] = 0.00

# plotting
param['plots_fn'] = 'plots.pdf'

# reference
param['states_name'] = [
	'Cart Position [m]',
	'Pole Angle [rad]',
	'Cart Velocity [m/s]',
	'Pole Velocity [rad/s]']
param['actions_name'] = [
	'Cart Acceleration [m/s^2]']

# reference trajectory
param['reference_trajectory'] = zeros((4,param.get('nt')))

# default system parameters
if param.get('system') is "CartPole":
	param['sys_n'] = 4
	param['sys_m'] = 1
	param['sys_actions'] = arange(-100,110,10,dtype = float) # Newtons applied at each timestep
	param['sys_card_A'] = len(param.get('sys_actions'))
	param['sys_mass_cart'] = 1.0
	param['sys_mass_pole'] = 0.1
	param['sys_length_pole'] = 0.5
	param['sys_init_state_bounds'] = array([1,0.5,1,1])
	param['sys_objective'] = 'track'
	param['sys_pos_bounds'] = 5
	param['sys_angle_bounds_deg'] = 60
	param['sys_g'] = 9.81