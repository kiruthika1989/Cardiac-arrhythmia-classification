classdef stacking_ensemble
    
    properties(Access = protected)
        
        model = {} % Trained meta (stacking) model
        learner = {} % Metal learner (callback)
        
        ensemble = {}; % Trained ensemble
    end
    
    methods
        
        % Class constructor
        function obj = stacking_ensemble(ensemble, learner)
            
            % Trained ensemble
            obj.ensemble = ensemble;
            
            % Meta learner
            obj.learner = learner;
        end
        
        % Fit stacking ensemble
        function obj = fit(obj, X, Y)
            
            % Predict models
            [~, x] = obj.ensemble.predict(X);
            
            % Train meta learner
            obj.model = obj.learner(x, Y);
                
        end
        
        % Predict new data
        function [Y] = predict(obj, X)
            
            % Predict with ensemble
            [~, x] = obj.ensemble.predict(X);
            
            % Predict with meta learner
            Y = obj.model.predict(x);
        end
    end
end

