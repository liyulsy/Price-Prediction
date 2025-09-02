def forward(self, x_enc, x_mark_enc, edge_index=None, edge_weight=None, news_features=None):
        B, S, N = x_enc.shape
        
        # --- UNIFIED ARCHITECTURE ---
        # 1. Multi-scale processing
        x_enc_scales, x_mark_enc_scales = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        
        # 2. Prepare data for TimeMixer
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, (x, x_mark) in enumerate(zip(x_enc_scales, x_mark_enc_scales)):
                B_s, T_s, N_s = x.size()
                # 跳过标准化，因为数据已在外部标准化
                # x = self.normalize_layers[i](x, 'norm')  # 注释掉避免双重标准化
                if self.configs.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B_s * N_s, T_s, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N_s, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in enumerate(x_enc_scales):
                B_s, T_s, N_s = x.size()
                # 跳过标准化，因为数据已在外部标准化
                # x = self.normalize_layers[i](x, 'norm')  # 注释掉避免双重标准化
                if self.configs.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B_s * N_s, T_s, 1)
                x_list.append(x)
        
        # 3. TimeMixer feature extraction
        features_list = self.timemixer(x_list, x_mark_list)

        # 4. Multi-scale feature processing
        dec_out_list = []
        for i, features_scale in enumerate(features_list):
            B_s, T_prime, d_model = features_scale.size()
            
            # Reshape to [B, N, T_prime, d_model]
            features_reshaped = features_scale.view(B, N, T_prime, d_model)
            features_for_downstream = features_reshaped

            # 5. Optional News Fusion
            if self.has_news:
                if news_features is None:
                    raise ValueError("Model initialized to use news, but no news_features were provided.")
                processed_news = self.news_processor(news_features)
                news_feat_expand = processed_news.unsqueeze(2).expand(-1, -1, T_prime, -1)
                features_for_downstream = torch.cat([features_reshaped, news_feat_expand], dim=-1)

            # 6. Optional GCN
            if self.use_gcn:
                if edge_index is None:
                    raise ValueError("Model initialized to use GCN, but no edge_index was provided.")
                
                # Reshape for GCN: [B, N, T_prime, d] -> [B*T_prime, N, d] -> [B*T_prime*N, d]
                gcn_input = features_for_downstream.permute(0, 2, 1, 3).reshape(B * T_prime * N, -1)
                # Create batch edge index
                num_graphs = B * T_prime
                edge_index_batch = edge_index.unsqueeze(0).repeat(num_graphs, 1, 1)
                offset = torch.arange(num_graphs, device=edge_index.device) * N
                edge_index_batch = edge_index_batch + offset.view(-1, 1, 1)
                edge_index_batch = edge_index_batch.view(2, -1)

                # Apply GCN
                if edge_weight is not None:
                    edge_weight_batch = edge_weight.repeat(B * T_prime)
                    gcn_out = self.gcn_layers[i](gcn_input, edge_index_batch, edge_weight_batch)
                else:
                    gcn_out = self.gcn_layers[i](gcn_input, edge_index_batch)

                features_for_downstream = gcn_out.view(B, T_prime, N, -1).permute(0, 2, 1, 3)
            
            # 7. Temporal aggregation with normalization
            aggregated_features = features_for_downstream.mean(dim=2)  # [B, N, d]
            
            # Add layer normalization to prevent feature collapse
            if hasattr(self, 'layer_norm'):
                aggregated_features = self.layer_norm(aggregated_features)
            
            aggregated_features_flat = aggregated_features.reshape(B * N, -1)
            
            # 8. Scale-specific prediction with dropout
            dec_out = self.predict_layers[i](aggregated_features_flat)
            dec_out = F.dropout(dec_out, p=0.2, training=self.training)  # 防止过拟合
            dec_out = dec_out.view(B, N, -1)
            dec_out_list.append(dec_out)
        
        # 9. Multi-scale fusion
        final_features = torch.stack(dec_out_list, dim=-1).sum(-1)  # [B, N, prediction_head_dim]
        
        # 10. Final MLP prediction
        final_features_flat = final_features.view(-1, final_features.size(-1))  # [B*N, prediction_head_dim]
        output_flat = self.mlp(final_features_flat)  # [B*N, num_classes]
        
        # 11. Reshape output based on task
        output = output_flat.view(B, N, -1)  # [B, N, num_classes]
        if self.task_type == 'regression':
            output = output.squeeze(-1)  # [B, N] for regression
        
        return output 