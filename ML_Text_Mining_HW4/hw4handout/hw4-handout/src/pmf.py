import numpy as np


class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies

    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):

        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]
        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            # TODO: implement your SGD here!!
            print(i,'th iteration')
            set_users = set(users)
            set_movies = set(movies)
            set_users_batch = set(users_batch)
            set_movies_batch = set(movies_batch)

            # i, j, k is needed to be defined for userID, movieID, and ratingID respectively.
            # i, j means the number of times in the user set, movie set,
            # k is the index of rating correspond to the userID and movieID.

            for user_id in list(set_users_batch):
                i = np.squeeze(np.where(list(set_users) == user_id)[0])

                for movie_id in list(set_movies_batch):
                    j = np.squeeze(np.where(list(set_movies) == movie_id)[0])
                    k = np.squeeze(np.where(np.logical_and((users == user_id), (movies == movie_id)))[0])

                    if k.size != 0:  # check if the rating is exist or not
                        dU = (np.dot(self.U[:, i].T, self.V[:, j]) - ratings[k]) * self.V[:, j] + lambda_u * self.U[:, i]
                        dV = (np.dot(self.V[:, i].T, self.V[:, j]) - ratings[k]) * self.U[:, i] + lambda_v * self.V[:, j]
                        self.U[:, i] = self.U[:, i] - alpha * dU
                        self.V[:, j] = self.V[:, j] - alpha * dV
