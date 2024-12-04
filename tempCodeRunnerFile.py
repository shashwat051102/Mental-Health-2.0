# Save the final model to a pickle file
# Save the best performing model to a pickle file
with open('final_model.pkl', 'wb') as pickle_out:
    pickle.dump(xgb_pipeline, pickle_out)  # or catboost_pipeline, depending on which performed better