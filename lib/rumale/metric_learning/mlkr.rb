# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/base/regressor'
require 'mopti/scaled_conjugate_gradient'

module Rumale
  module MetricLearning
    # MLKR is a class that implements Metric Learning for Kernel Regression.
    #
    # @example
    #   transformer = Rumale::MetricLearning::MLKR.new
    #   transformer.fit(training_samples, traininig_labels)
    #   low_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Weinberger, K. Q. and Tesauro, G., "Metric Learning for Kernel Regression," Proc. AISTATS'07, pp. 612--629, 2007.
    class MLKR
      include Base::BaseEstimator
      include Base::Transformer
      include Base::Regressor

      # Return the prototypes for the regressor.
      # @return [Numo::DFloat] (shape: [n_training_samples, n_components])
      attr_reader :prototypes

      # Return the values of the prototypes
      # @return [Numo::DFloat] (shape: [n_training_samples])
      attr_reader :values

      # Returns the neighbourhood components.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with MLKR.
      #
      # @param n_components [Integer] The number of components.
      # @param init [String] The initialization method for components ('random' or 'pca').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: nil, init: 'random', max_iter: 100, tol: 1e-6, verbose: false, random_seed: nil)
        check_params_numeric_or_nil(n_components: n_components, random_seed: random_seed)
        check_params_numeric(max_iter: max_iter, tol: tol)
        check_params_string(init: init)
        check_params_boolean(verbose: verbose)
        @params = {}
        @params[:n_components] = n_components
        @params[:init] = init
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @components = nil
        @n_iter = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [NeighbourhoodComponentAnalysis] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_narray(y)
        check_sample_label_size(x, y)
        n_features = x.shape[1]
        n_components = if @params[:n_components].nil?
                         n_features
                       else
                         [n_features, @params[:n_components]].min
                       end
        @components, @n_iter = optimize_components(x, y, n_features, n_components)
        @prototypes = x.dot(@components.transpose)
        @values = y
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_narray(y)
        check_sample_label_size(x, y)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = check_convert_sample_array(x)
        x.dot(@components.transpose)
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples]) Predicted values per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        z = transform(x)
        kernel_mat = Rumale::PairwiseMetric.rbf_kernel(z, @prototypes)
        kernel_mat.dot(@values) / kernel_mat.sum(1)
      end

      private

      def check_convert_narray(y)
        y = Numo::NArray.cast(y) unless y.is_a?(Numo::NArray)
        raise ArgumentError, 'Expect label/target vector to be 1-D arrray' unless y.ndim == 1
        y
      end

      def init_components(x, n_features, n_components)
        if @params[:init] == 'pca'
          pca = Rumale::Decomposition::PCA.new(n_components: n_components, solver: 'evd')
          pca.fit(x).components.flatten.dup
        else
          Rumale::Utils.rand_normal([n_features, n_components], @rng.dup).flatten.dup
        end
      end

      def optimize_components(x, y, n_features, n_components)
        # initialize components.
        comp_init = init_components(x, n_features, n_components)
        # initialize optimization results.
        res = {}
        res[:x] = comp_init
        res[:n_iter] = 0
        # perform optimization.
        optimizer = Mopti::ScaledConjugateGradient.new(
          fnc: method(:mlkr_loss), jcb: method(:mlkr_dloss),
          x_init: comp_init, args: [x, y],
          max_iter: @params[:max_iter], ftol: 1e-16#@params[:tol]
        )
        fold = 0.0
        dold = 0.0
        optimizer.each do |prm|
          res = prm
          puts "[MLKR] The value of loss function after #{res[:n_iter]} epochs: #{res[:fnc]}" if @params[:verbose]
          break if (fold - res[:fnc]).abs <= @params[:tol] && (dold - res[:jcb]).abs <= @params[:tol]
          fold = res[:fnc]
          dold = res[:jcb]
        end
        # return the results.
        n_iter = res[:n_iter]
        comps = n_components == 1 ? res[:x].dup : res[:x].reshape(n_components, n_features)
        [comps, n_iter]
      end

      def mlkr_loss(w, x, y)
        # initialize some variables.
        n_samples, n_features = x.shape
        n_components = w.size / n_features
        # projection.
        w = w.reshape(n_components, n_features)
        z = x.dot(w.transpose)
        # predict values.
        kernel_mat = Rumale::PairwiseMetric.rbf_kernel(z)
        tmp = kernel_mat - kernel_mat[kernel_mat.diag_indices].diag
        y_pred = tmp.dot(y) / tmp.sum(1)
        # calculate loss.
        ((y_pred - y)**2).sum
      end

      def mlkr_dloss(w, x, y)
        # initialize some variables.
        n_features = x.shape[1]
        n_components = w.size / n_features
        # projection.
        w = w.reshape(n_components, n_features)
        z = x.dot(w.transpose)
        # predict values.
        kernel_mat = Rumale::PairwiseMetric.rbf_kernel(z)
        tmp = kernel_mat - kernel_mat[kernel_mat.diag_indices].diag
        y_pred = tmp.dot(y) / tmp.sum(1)
        # calculate gradient.
        n_samples = x.shape[0]
        gradient = Numo::DFloat.zeros(n_components, n_features)
        n_samples.times do |i|
          xx = (x[i, true] - x)
          zz = (z[i, true] - z)
          wi = kernel_mat[i, true] * (y_pred - y)
          gradient += (y_pred[i] - y[i]) * (zz.transpose * wi).dot(xx)
        end
        (4 * gradient).flatten.dup
      end
    end
  end
end
