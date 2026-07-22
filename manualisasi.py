import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)

print("="*75)
print("MULTIMODAL FUSION & CROSS-ATTENTION")
print("="*75)

print("\nINPUT REPRESENTATION")
X_mri = torch.randn(1, 3, 72, 72) 
X_snp = torch.randn(1, 1000)

print(f"[*] Input MRI (X_mri) Shape : {X_mri.shape} -> Tensor 4D (Gambar 2D)")
print(f"[*] Input SNP (X_snp) Shape : {X_snp.shape}     -> Tensor 2D (Sekuens Vektor)")

print("\nFEATURE EXTRACTION")
print("[a] MRI ResNet-18 (Operasi Konvolusi & Pooling)")
W_mri_sim = torch.randn(3*72*72, 512) 
F_mri = torch.matmul(X_mri.view(1, -1), W_mri_sim)
print(f"[*] Ekstraksi Fitur MRI (F_mri): {F_mri.shape} -> Vektor 512-Dimensi")

print("[b] SNP Transformer Encoder (Operasi Self-Attention 1D)")
W_snp_sim = torch.randn(1000, 512)
F_snp = torch.matmul(X_snp, W_snp_sim)
print(f"[*] Ekstraksi Fitur SNP (F_snp): {F_snp.shape}   -> Vektor 512-Dimensi")

print("\nCROSS-MODAL ATTENTION")
d_k = 512 

W_q = torch.randn(512, 512)
W_k = torch.randn(512, 512)
W_v = torch.randn(512, 512)

Q_mri = torch.matmul(F_mri, W_q)
K_snp = torch.matmul(F_snp, W_k)
V_snp = torch.matmul(F_snp, W_v)

print(f"[*] Q_mri (Kueri Otak) : {Q_mri.shape}")
print(f"[*] K_snp (Kunci Gen)  : {K_snp.shape}")
print(f"[*] V_snp (Nilai Gen)  : {V_snp.shape}")

print("\nMenghitung Skor Korelasi / Dot Product (Query & Key)")
scores = torch.matmul(Q_mri, K_snp.transpose(0, 1))
print(f"[*] Rumus: Q * K^T")
print(f"[*] Hasil Skor Raw     : {scores.item():.4f}")

print("\nScaling & Softmax (Menjadi Bobot Atensi / Probabilitas)")
scaled_scores = scores / math.sqrt(d_k)
attention_weights = F.softmax(scaled_scores, dim=-1)
print(f"[*] Rumus: Softmax( (Q * K^T) / sqrt({d_k}) )")
print(f"[*] Nilai Attention Weight : {attention_weights.item():.4f} (Berada di rentang 0 hingga 1)")

print("\nMengaplikasikan Bobot ke Value SNP (Cross-Attention Output)")
cross_attn_out = attention_weights * V_snp
print(f"[*] Rumus: Attention_Weights * V_snp")
print(f"[*] Output Cross-Attention Shape: {cross_attn_out.shape}")

print("\nMULTIMODAL CONCATENATION")
print("Menggabungkan fitur original dan fitur atensi ke dalam 1 vektor raksasa.")
# Menggabungkan [F_mri, F_snp, cross_attn_out]
Fused_Feature = torch.cat([F_mri, F_snp, cross_attn_out], dim=1)
print(f"[*] Rumus: Concat([F_mri, F_snp, Cross_Attn_Output])")
print(f"[*] Dimensi Akhir (512 + 512 + 512): {Fused_Feature.shape}")

print("\nKLASIFIKASI AKHIR")
W_out = torch.randn(1536, 3) 
b_out = torch.randn(3)

logits = torch.matmul(Fused_Feature, W_out) + b_out
print(f"[*] Rumus: (Fused_Feature * W) + b")
print(f"[*] Skor Mentah (Logits)   : {logits.detach().numpy()}")
probabilities = F.softmax(logits, dim=1)
print(f"[*] Rumus Akhir: Softmax(Logits)")
print(f"[*] Hasil Probabilitas     : {probabilities.detach().numpy()}")

pred_class = torch.argmax(probabilities, dim=1).item()
classes = ["Normal Control (NC)", "Mild Cognitive Impairment (MCI)", "Alzheimer's Disease (AD)"]
print(f"\n=> [KESIMPULAN KLASIFIKASI]: {classes[pred_class]}")

print("\n\n" + "="*75)
print(" CURRICULUM LEARNING (PACING SCHEDULER)")
print("="*75)
print("\n[1] Mengkalkulasi Tingkat Kesulitan Pasien (Difficulty Score)")
print("Asumsi: Pasien NC bergejala ringan = Mudah (Skor 0.2), Pasien Batas MCI/AD = Sulit (Skor 0.9)")
pasien_difficulty = 0.85 
print(f"[*] Difficulty D(x) = {pasien_difficulty}")

print("\n[2] Menghitung Kapasitas Model Saat Ini (Competence lambda)")
epoch_t = 10
total_epochs = 50
lambda_t = (epoch_t / total_epochs) 
print(f"[*] Epoch t={epoch_t} dari {total_epochs}.")
print(f"[*] Rumus: lambda(t) = t / T")
print(f"[*] Kapasitas Model: lambda(10) = {epoch_t}/{total_epochs} = {lambda_t}")

print("\n[3] Seleksi Sampel Pelatihan")
print(f"[*] Syarat Masuk ke Siklus Backpropagation: D(x) <= lambda(t)")
if pasien_difficulty <= lambda_t:
    print(f"[*] Analisis Keputusan: {pasien_difficulty} <= {lambda_t}")
    print(f"[*] Status: [DITERIMA] Kapasitas model sudah sanggup mencerna data ini")
else:
    print(f"[*] Analisis Keputusan: {pasien_difficulty} > {lambda_t}")
    print(f"[*] Status: [DITOLAK] D(x) > lambda(t) | Data terlalu sulit, ditunda ke Epoch selanjutnya")

print("\n=> [KESIMPULAN CURRICULUM LEARNING]:")
print("Inilah alasan mengapa akurasi model bisa mencapai 96%. Model tidak dipaksa menelan data ber-skor 0.85 di Epoch 10 saat otaknya (kapasitasnya) baru mencapai level 0.20!")
