import numpy as np


class TestRotation:
	def rot1(self):
		origin = np.array([0, 0, 0])

		vec1 = np.array([-0.58238, 0.58456, 0.56491])
		vec2 = np.array([-0.37844, 0.42006, -0.82482])
		vecnorm = np.cross(vec1, vec2)
		inverted_vecnorm = np.negative(vecnorm)
		P = np.array([origin, inverted_vecnorm])

		refvec1 = np.array([0.75254023, -0.65807384, 0.02493937])
		refvec2 = np.array([0.00757324, -0.02921987, -0.99954432])
		refnorm = np.cross(refvec1, refvec2)
		Q = np.array([origin, refnorm])

		M = Q.dot(np.linalg.pinv(P))
		print(f'refnorm: {refnorm}')
		print(f'vecnorm: {inverted_vecnorm}')

		# print(f'rotated vec1: {M.dot(np.array([origin, vec1]))}')
		print(f'rotated vec1:\n{np.array2string(M.dot(P)[1], separator=", ")}')
		print(f'Q:\n{np.array2string(Q[1], separator=", ")}')

	def rot2(self):
		origin = np.array([0, 0, 0])

		vec1 = np.array([-0.58238, 0.58456, 0.56491])
		vec2 = np.array([-0.37844, 0.42006, -0.82482])
		P = np.array([vec1, origin, vec2])

		refvec1 = np.array([0.75254023, -0.65807384, 0.02493937])
		refvec2 = np.array([0.00757324, -0.02921987, -0.99954432])
		Q = np.array([refvec1, origin, refvec2])

		P_inv = np.linalg.pinv(P)
		M = Q.dot(P_inv)
		print(f'Calculated Q:\n{M.dot(P)}')
