classdef MinMaxScaler < BaseEstimator & TransformerMixin
    % Transforms features by scaling each feature to a given range.
    %
    % This estimator scales and translates each feature individually such
    % that it is in the given range on the training set.
    
    properties (GetAccess = 'public', SetAccess = 'public')
        % parameters
        feature_range = [0 1];
    end
    
    properties (GetAccess = 'public', SetAccess = 'private')
        % attributes
        scale_; % Per feature relative scaling of the data.
        data_min_; % Per feature minimum seen in the data.
        data_max_; % Per feature maximum seen in the data.
        data_range_; % Per feature range seen in the data.
    end
    
    methods
        % constructor
        function obj = MinMaxScaler(params)
            if nargin>0
                obj.set_params(params)
            end
        end
        
        % Compute the minimum and maximum to be used for later scaling.
        function fit(obj,X,~)
            obj.data_min_ = min(X,[],1);
            obj.data_max_ = max(X,[],1);
            obj.data_range_ = obj.data_max_ - obj.data_min_;
           
            % handle data range for zero variance data
            obj.data_range_(obj.data_range_==0) = 1;
                
            obj.scale_ = bsxfun(@rdivide,diff(obj.feature_range),...
                obj.data_range_);
        end
        
        % Scaling features of X according to feature_range.
        function X_new = transform(obj,X)
            % scale between 0 and 1
            X_new = bsxfun(@minus,X,obj.data_min_);
            X_new = bsxfun(@rdivide,X_new,obj.data_range_);
            
            % scale between min and max
            X_new = bsxfun(@times,X_new,diff(obj.feature_range));
            X_new = bsxfun(@plus,X_new,obj.feature_range(1));
        end
        
        % Undo the scaling of X according to feature_range.
        function X_orig = inverse_transform(obj,X)
            % scale between 0 and 1
            X_orig = (X - obj.feature_range(1))/diff(obj.feature_range);
            
            % scale between data_min_ and data_max_
            X_orig = bsxfun(@times,X_orig,obj.data_max_-obj.data_min_);
            X_orig = bsxfun(@plus,X_orig,obj.data_min_);
        end
    end
end
