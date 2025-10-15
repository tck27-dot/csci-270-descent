#from algorithms import momentum_grad_descent
#from levels import Level2, visualize_journey
#env = Level2()
#points = momentum_grad_descent(.95,env)
#visualize_journey(points,env.can_see_exit(points[-1]))



from algorithms import momentum_grad_descent
from algorithms import vanilla_grad_descent
from levels import Level3, visualize_journey, Level4
#env = Level3()
env = Level4()
#points = momentum_grad_descent(.95,env)
points = vanilla_grad_descent(.05,env)
visualize_journey(points,env.can_see_exit(points[-1]))


