import torch
import torch.nn as nn
import torch.nn.functional as F
from ..BaseModel.cnn import CNN 
from ..BaseModel.gcn import GCN

class CnnGnnNoNews(nn.Module):
    def __init__(self, 
                 price_seq_len, 
                 num_nodes, 
                 cnn_output_channels=32, 
                 gcn_hidden_dim=128, 
                 gcn_output_dim=64, 
                 final_mlp_hidden_dim=128, 
                 num_classes=2):
        super().__init__()
        self.price_seq_len = price_seq_len
        self.num_nodes = num_nodes
        self.price_cnn = CNN(in_dim=price_seq_len, num_nodes=num_nodes, out_channels=cnn_output_channels)
        self.gcn_input_dim = cnn_output_channels
        self.gcn = GCN(input_dim=self.gcn_input_dim, 
                       hidden_dim=gcn_hidden_dim, 
                       output_dim=gcn_output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(gcn_output_dim, final_mlp_hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(final_mlp_hidden_dim, num_classes) 
        )
        
    # def forward(self, price_data_x, edge_index):
    #     B, S, N = price_data_x.shape
    #     if price_data_x.dim() == 3 and price_data_x.shape[2] == self.num_nodes:
    #          price_data_x_for_cnn = price_data_x.permute(0, 2, 1)
    #     elif price_data_x.dim() == 3 and price_data_x.shape[1] == self.num_nodes:
    #          price_data_x_for_cnn = price_data_x
    #     else:
    #         raise ValueError(f"CnnGnnNoNews: price_data_x 输入形状 {price_data_x.shape} 不符合预期 ([B,S,N] 或 [B,N,S])")

    #     cnn_node_features = self.price_cnn(price_data_x_for_cnn)
        
    #     if (cnn_node_features.shape[0] != B or
    #         cnn_node_features.shape[1] != self.num_nodes or
    #         cnn_node_features.shape[2] != self.gcn_input_dim):
    #         raise ValueError(
    #             f"CnnGnnNoNews: CNN输出形状 {cnn_node_features.shape} 与GCN输入维度预期不符。" + 
    #             f"CNN输出应为 [B, NumNodes, GCN_input_dim ({self.gcn.input_dim})]. " + 
    #             f"请检查CNN模型 ({type(self.price_cnn).__name__}) 的实现和out_channels参数。"
    #         )

    #     fused_node_features = cnn_node_features
        
    #     # # GCN处理部分（已注释）
    #     # fused_node_features_flat = fused_node_features.reshape(B * N, -1)
    #     # edge_index_batch_list = []
    #     # for i in range(B):
    #     #     offset = i * N
    #     #     edge_index_batch_list.append(edge_index + offset)
    #     # edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)
    #     # gcn_out = self.gcn(fused_node_features_flat, edge_index_for_gcn)  # [B * N, gcn_out_dim]
    #     # gcn_out = gcn_out.view(B, N, -1)
        
    #     # 直接使用CNN特征进行预测
    #     out = self.mlp(fused_node_features)  # [B, N, num_classes]

    #     return out
    

    def forward(self, price_data_x, edge_index):
        B, S, N = price_data_x.shape
        if price_data_x.dim() == 3 and price_data_x.shape[2] == self.num_nodes:
             price_data_x_for_cnn = price_data_x.permute(0, 2, 1)
        elif price_data_x.dim() == 3 and price_data_x.shape[1] == self.num_nodes:
             price_data_x_for_cnn = price_data_x
        else:
            raise ValueError(f"CnnGnnNoNews: price_data_x 输入形状 {price_data_x.shape} 不符合预期 ([B,S,N] 或 [B,N,S])")

        cnn_node_features = self.price_cnn(price_data_x_for_cnn)
        
        if (cnn_node_features.shape[0] != B or
            cnn_node_features.shape[1] != self.num_nodes or
            cnn_node_features.shape[2] != self.gcn_input_dim):
            raise ValueError(
                f"CnnGnnNoNews: CNN输出形状 {cnn_node_features.shape} 与GCN输入维度预期不符。" + 
                f"CNN输出应为 [B, NumNodes, GCN_input_dim ({self.gcn.input_dim})]. " + 
                f"请检查CNN模型 ({type(self.price_cnn).__name__}) 的实现和out_channels参数。"
            )

        fused_node_features = cnn_node_features
        fused_node_features_flat = fused_node_features.reshape(B * N, -1)

        edge_index_batch_list = []
        for i in range(B):
            offset = i * N
            edge_index_batch_list.append(edge_index + offset)
        edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)

        gcn_out = self.gcn(fused_node_features_flat, edge_index_for_gcn)  # [B * N, gcn_out_dim]
        gcn_out = gcn_out.view(B, N, -1)
        out = self.mlp(gcn_out)  # [B, N, num_classes]

        return out
    
    

if __name__ == '__main__':
    PRICE_SEQ_LEN_TEST = 180      
    NUM_NODES_TEST = 8            
    CNN_OUT_CHANNELS_TEST = 32
    GCN_HIDDEN_DIM_TEST = 64
    GCN_OUTPUT_DIM_TEST = 32      
    FINAL_MLP_HIDDEN_DIM_TEST = 128
    NUM_CLASSES_TEST = 2
    BATCH_SIZE_TEST = 4
    model_no_news = CnnGnnNoNews(
        price_seq_len=PRICE_SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        cnn_output_channels=CNN_OUT_CHANNELS_TEST,
        gcn_hidden_dim=GCN_HIDDEN_DIM_TEST,
        gcn_output_dim=GCN_OUTPUT_DIM_TEST,
        final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM_TEST,
        num_classes=NUM_CLASSES_TEST
    )
    model_no_news.train()
    print("\n--- CnnGnnNoNews 模型结构 ---")
    print(model_no_news)
    dummy_price_data = torch.randn(BATCH_SIZE_TEST, PRICE_SEQ_LEN_TEST, NUM_NODES_TEST)
    edge_list_test = []
    for i in range(NUM_NODES_TEST):
        for j in range(NUM_NODES_TEST):
            if i != j:
                edge_list_test.append([i, j])
    dummy_edge_index_test = torch.tensor(edge_list_test, dtype=torch.long).t().contiguous()
    print(f"\n--- 喂入虚拟数据到 CnnGnnNoNews 模型 ---")
    print(f"虚拟价格数据 (price_data_x) shape: {dummy_price_data.shape}")
    print(f"虚拟边索引 (edge_index) shape: {dummy_edge_index_test.shape}")
    try:
        output_logits_no_news = model_no_news(dummy_price_data, dummy_edge_index_test)
        print(f"\n模型输出 (output_logits_no_news) shape: {output_logits_no_news.shape}")
        expected_output_shape = (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
        assert output_logits_no_news.shape == expected_output_shape, \
            f"输出形状不匹配! 得到: {output_logits_no_news.shape}, 期望: {expected_output_shape}"
        print(f"模型输出 (第一个样本，第一个节点): {output_logits_no_news[0, 0, :]}")
        print("\nCnnGnnNoNews 模型测试成功!")
    except Exception as e:
        print(f"\nCnnGnnNoNews 模型测试失败: {e}")
        import traceback
        traceback.print_exc()