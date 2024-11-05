# Evaluating Our Model Using nDDG Metric
# Function in TrainTestUtilities.py
def evaluate_nddg(QNS_list, model, transform, device='cpu'):
    final_order = []
    rev_orders = []
    model.eval()
    with torch.no_grad():  # No need to track gradients during evaluation
        for q_element in QNS_list:
            fss = []
            rss = []
            # Load and transform the query image
            query_tensor = transform(q_element.query_vector).unsqueeze(0).to(device)
            vec_ref = model(query_tensor)
            count = 0
            sz = len(q_element.neighbor_vectors)
            for neighbor_path in q_element.neighbor_vectors:
                # Load and transform the neighbor image
                neighbor_tensor = transform(neighbor_path).unsqueeze(0).to(device)
                vec_i = model(neighbor_tensor)

                distf = torch.norm(vec_ref - vec_i)
                fss.append(distf.item())
                rss.append(sz-count)
                count +=1
            final_order.append(fss) 
            rev_orders.append(rss)
    model_acc = 100 * np.mean((test_ndcg(final_order) -  test_ndcg(rev_orders))/(1 -test_ndcg(rev_orders)))
    return model_acc, final_order


#Euclidean Evaluate - Function in TabularUtilities.py
def euclidian_evaluate(QNS_list, iweights, is_mat, dim=0):
    
    # acc_base, ddg_base = euclidian_base(QNS_list)
    # print(f'The Dataset Base Accuracy: {acc_base:.4} and DDG: {ddg_base:.4}!')

    qns = copy.deepcopy(QNS_list)
    # Preparing Testset into right format
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()
    
    # Preparing the weights into right format
    weights = torch.from_numpy(iweights)
    weights = weights.type(torch.float64)

    # Accuracy Evaluation
    final_ordering = []
    rev_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        rs_dist = []
        sz = len(q_element.neighbor_vectors)
        count = 0
        for i in range(q_element.neighbor_count):
            dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1
               
            qn_dist.append(dist_i.item())
            rs_dist.append(sz - count)
            count +=1
        final_ordering.append(qn_dist)
        rev_ordering.append(rs_dist)
    
    acc = success_count/total_count
    #ddg = 100 * numpy.mean(test_ndcg(final_ordering))
    ddg = 100 * numpy.mean((test_ndcg(final_ordering) -  test_ndcg(rev_ordering))/(1 -test_ndcg(rev_ordering)))

    # print(f'Acc-Current: {acc:.4} ({100*(acc-acc_base)/acc_base:.4}%) | DDG-Current: {ddg:.4} ({100*(ddg-ddg_base)/ddg_base:.4}%)')
    return acc, ddg
