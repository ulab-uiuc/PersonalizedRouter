import random
import numpy as np
import torch
from graph_nn_new_v2 import  form_data,GNN_prediction
from data_processing.utils import savejson,loadjson,savepkl,loadpkl
import pandas as pd
import json
import re
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"


class graph_router_prediction:
    def __init__(self, router_data_path,llm_path,llm_embedding_path,config,wandb):
        self.config = config
        self.wandb = wandb
        self.data_df = pd.read_csv(router_data_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms=len(self.llm_names)
        self.num_users = config['user_num']
        self.num_unique_query=int(len(self.data_df) / (self.num_llms * self.num_users))
        self.num_task=config['num_task']
        self.set_seed(self.config['seed'])
        self.llm_description_embedding=loadpkl(llm_embedding_path)
        self.query_mapping, self.llm_mapping = self.create_mapping(router_data_path)
        self.prepare_data_for_GNN()
        self.split_data()
        self.form_data = form_data(device)
        self.query_dim = self.query_embedding_list.shape[1]
        self.llm_dim = self.llm_description_embedding.shape[1]
        self.user_dim = self.user_embedding.shape[1]
        self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim, user_feature_dim=self.user_dim,
                                    hidden_features_size=self.config['embedding_dim'], in_edges_size=self.config['edge_dim'],wandb=self.wandb,config=self.config,device=device)
        print("GNN training successfully initialized.")
        self.train_GNN()
        # self.test_GNN()

    def create_mapping(self,file_path):
        df = pd.read_csv(file_path)
        unique_query = df['query'].drop_duplicates().tolist()
        query_mapping = {query: idx for idx, query in enumerate(unique_query)}
        unique_llm = df['llm'].drop_duplicates().tolist()
        llm_mapping = {llm: idx for idx, llm in enumerate(unique_llm)}
        return query_mapping, llm_mapping


    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def split_data(self):
        self.query_per_user = int(len(self.data_df) / self.num_users)
        self.unique_query_per_user = int(len(self.data_df) / self.num_users / self.num_task)
        split_ratio = self.config['split_ratio']

        train_size = int(self.unique_query_per_user * split_ratio[0])
        val_size = int(self.unique_query_per_user * split_ratio[1])
        test_size = int(self.unique_query_per_user * split_ratio[2])

        train_idx = []
        validate_idx = []
        test_idx = []

        for user_id in range(self.num_users):
            for task_id in range(self.num_task):
                start_idx = user_id * self.query_per_user + task_id * self.unique_query_per_user

                train_idx.extend(range(start_idx, start_idx + train_size))

                validate_idx.extend(range(start_idx + train_size,
                                          start_idx + train_size + val_size))
                test_idx.extend(range(start_idx + train_size + val_size,
                                      start_idx + train_size + val_size + test_size))




        self.combined_edge=np.concatenate((self.cost_list.reshape(-1,1),self.effect_list.reshape(-1,1)),axis=1)
        for i, (effect, cost) in enumerate(zip( self.effect_list, self.cost_list)):
            self.effect_list[i] = 0.5 * effect - (1 - 0.5) * cost

        self.label = np.array(self.data_df['best_answer'].tolist()).reshape(-1, 1)
        self.edge_org_id = [self.query_mapping[query] for query in self.query_list]
        self.edge_des_id= list(range(self.edge_org_id[0],self.edge_org_id[0]+self.num_llms)) * int(self.num_unique_query * self.num_users )

        self.mask_train =torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx]=1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1


    def prepare_data_for_GNN(self):
        query_embedding_list_raw=self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()
        self.query_list=np.array(self.data_df['query'].tolist())
        self.query_embedding_list= []
        self.task_embedding_list= []
        for inter in query_embedding_list_raw:
            inter=re.sub(r'\s+', ', ', inter.strip())
            try:
                inter=json.loads(inter)
            except:
                inter = inter.replace("[[,", "[[")
                inter = json.loads(inter)
            self.query_embedding_list.append(inter[0])

        for inter in task_embedding_list_raw:
            inter=re.sub(r'\s+', ', ', inter.strip())
            try:
                inter=json.loads(inter)
            except:
                inter = inter.replace("[[,", "[[")
                inter = json.loads(inter)
            self.task_embedding_list.append(inter[0])

        unique_dict = {}
        for idx, item in enumerate(self.query_list):
            item_tuple = tuple(item)
            if item_tuple not in unique_dict:
                unique_dict[item_tuple] = idx
        unique_query_embedding_list = np.array([self.query_embedding_list[idx] for idx in unique_dict.values()])
        unique_task_embedding_list = np.array(
            [self.task_embedding_list[idx] for idx in unique_dict.values()])
        self.query_embedding_list = unique_query_embedding_list
        self.task_embedding_list = unique_task_embedding_list

        self.user_embedding= np.eye(self.num_users)

        self.effect_list=np.array(self.data_df['effect'].tolist())
        self.cost_list=np.array(self.data_df['cost'].tolist())
        self.answer_list = np.array(self.data_df['best_answer'].tolist())




    def train_GNN(self):

        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             user_feature=self.user_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.effect_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test,
                                                             llm_mapping=self.llm_mapping,
                                                             cost_list=self.cost_list,
                                                             answer_list=self.answer_list,
                                                             config = self.config)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                user_feature=self.user_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.effect_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test,
                                                                llm_mapping=self.llm_mapping,
                                                                cost_list=self.cost_list,
                                                                answer_list=self.answer_list,
                                                                config=self.config)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        user_feature=self.user_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test,
                                                        llm_mapping=self.llm_mapping,
                                                        cost_list = self.cost_list,
                                                        answer_list=self.answer_list,
                                                        config=self.config)
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,data_for_test=self.data_for_test)

    def test_GNN(self):
        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        user_feature=self.user_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test,
                                                        llm_mapping=self.llm_mapping,
                                                        cost_list = self.cost_list,
                                                        answer_list=self.answer_list,
                                                        config=self.config)
        predicted_result = self.GNN_predict.test(data=self.data_for_test,model_path=self.config['model_path'])
        print(predicted_result)




if __name__ == "__main__":
    import wandb
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="graph_router")
    graph_router_prediction(router_data_path=config['saved_router_data_path'],llm_path=config['llm_description_path'],
                            llm_embedding_path=config['llm_embedding_path'],config=config,wandb=wandb)