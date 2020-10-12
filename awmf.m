fid=fopen('ansoffwmf.txt','wt');
ch=5;
for ii=4:4
  for jj=3:3
    for kk=2:2
      for oo=3:3
        for zz=2:2
        
randn('seed',1.0);   
rand('seed',1.0);

kkn=[10,20,30,40,50,60,70,80,90,100]
k=kkn(kk);

data=load('trainningdata.txt');
test=load('testdata.txt');

%Each line of trainingdata.txt is: UserID \t ItemID \t 1
%Each line of testdata.txt is :UserID \t ItemID \t 1

a=max([data;test],[],1);
n=a(1);
m=a(2);
lambdan=[0.5,0.1,0.01,0.001,0.0001];
dd=[10,20,30,50,100];
lambda=0.01;
alpha=0.1;
alphan=[0.5,0.1,0.05,0.01]
alpha1=alphan(zz);
eron=[1e-4,1e-5,1e-6];
ero=1e-5;
etan=[0.1,1,5];
eta=1;
lambdag=lambdan(ii);
lambda1=lambdan(jj);
lambda2=lambdan(oo);
d=20;
r1=data(:,1);
r2=data(:,2);

nn=length(r1); 
HR=sparse(r1,r2,ones(nn,1),n,m);
nep=1000;
idcg=zeros(n,1);
idcg1=zeros(n,1);
idcg2=zeros(n,1);

     tt=length(test(:,1));
    H=sparse(test(:,1),test(:,2),ones(tt,1),n,m);
 
    num=sum(H,2);
 
    
   
    for i=1:n 
        [pp,an]=sort(H(i,:),2,'descend');
        for j=1:min(ch,num(i))
            idcg(i)=idcg(i)+1/log2(j+1);
        end
         for j=1:num(i)
            idcg1(i)=idcg1(i)+1/log2(j+1);
        end
      
    end
  
bestndcgnum=zeros(m,1);
bestndcgnum(1)=1;
for i=2:m
  bestndcgnum(i)=bestndcgnum(i-1)+1/log2(i+1);
end
    
  u=rand(n,k)*0.1;
  v=rand(m,k)*0.1;
  g=(rand(n,d)-0.5);
  w1=(rand(m,1)*0.01+0.01);
  w2=(rand(m,1)-0.5);
  h=[w1;w2];
   p=exp(g)./sum(exp(g),2);
   sx=(p'*HR)';
   q=ga(w1.*sx+w2);

    negconst=5;
    
opadam=cell(2,2);
  for i=1:4
      opadam{i,1}=zeros(size(u));
      opadam{i,2}=zeros(size(v));
      opadam{i,3}=zeros(size(g));
      opadam{i,4}=zeros(size(h));
   end





itempop=sum(HR,1);
tranx=sparse(r1,1:nn,ones(nn,1),n,nn);
trany=sparse(r2,1:nn,ones(nn,1),m,nn);

    premax=-1000;
    remax=-1000;
    ndmax=-1000;

% update    
for ep=1:nep


    % calculate gradient
    zhongqv=tendot(q,v);
    mu=zhongqv'*v;
    zhongpu=tendot(p,u);
    posw=sum(p(r1,:).*q(r2,:),2);
    zhongposv=posw.*v(r2,:);
    du=2*(zhongpu*mu-tranx*zhongposv);
    du=du+lambda*u;

    zhongpu=tendot(p,u);
    mv=zhongpu'*u;
    zhongqv=tendot(q,v);
    posw=sum(p(r1,:).*q(r2,:),2);
    zhongposu=posw.*u(r1,:);
    dv=2*(zhongqv*mv-trany*zhongposu);
    dv=dv+lambda*v;

    zhongvv=tendot(v,v);
    mp=zhongvv'*q;
    sp=sum(q,1);
    zhonguu=tendot(u,u);
    pospre=sum(u(r1,:).*v(r2,:),2);
    zhongposp=(-2*pospre+2*eta*ero-(eta-1)).*q(r2,:);
    dp1=zhonguu*mp-eta*ero.^2*sp+tranx*zhongposp;
   


    zhonguu=tendot(u,u);
    mq=zhonguu'*p;
    sq=sum(p,1);
    zhongvv=tendot(v,v);
    pospre=sum(u(r1,:).*v(r2,:),2);
    zhongposq=(-2*pospre+2*eta*ero-(eta-1)).*p(r1,:);
    dq=zhongvv*mq-eta*ero.^2*sq+trany*zhongposq;
    dw1=sum(dq.*q.*(1-q).*sx,2);
    dw2=sum(dq.*q.*(1-q),2);
    dw1=dw1+lambda1*w1;
    dw2=dw2+lambda2*w2;
    dh=[dw1;dw2];
    dsx=dq.*q.*(1-q).*w1;
    dp2=(dsx'*HR')';
    dp=dp1+dp2;
    sumdp=sum(dp.*p,2);
    dg=p.*(-sumdp+dp);
    dg=dg+lambdag*g;

    

  

    % update with ADAM
    prga={du,dv,dg,dh};
    % adam
    ll=length(prga);
    beta1=0.9;
    beta2=0.999;
    eps=1e-8;
    nowga=cell(1,ll);
    for i=1:ll
      opadam{1,i}=opadam{1,i}*beta1+(1-beta1)*prga{1,i};
      opadam{2,i}=opadam{2,i}*beta2+(1-beta2)*(prga{1,i}.^2);
      nowm=opadam{1,i}/(1-beta1.^ep);
      nowv=opadam{2,i}/(1-beta2.^ep);
      nowga{i}=nowm./(sqrt(nowv)+eps);
    end

    u=u-alpha*nowga{1};
    v=v-alpha*nowga{2};
    g=g-alpha1*nowga{3};
    h=h-alpha1*nowga{4};

    w1=max(1e-5,h(1:m,:));
    w2=h(m+1:end,:);

   p=exp(g)./sum(exp(g),2);
   sx=(p'*HR)';
   q=ga(w1.*sx+w2);
    


   % evaluate

   %S1: prediction by u*v'
     if(ep>=1&&mod(ep,25)==0)
        R=u*v';
        R(sub2ind(size(R),r1,r2))=-inf;
        [~,an]=sort(R,2,'descend');

        presion=zeros(n,1);
        recall=zeros(n,1);
         ndcg=zeros(n,1);
         ndcg1=zeros(n,1);
         mrr=zeros(n,1);
          num=sum(H,2);
        cao=full(sum(num~=0));                                       
         for i=1:n
           id1=find(H(i,:)~=0);
           id2=an(i,1:ch);
           pr=intersect(id1,id2);
           if(num(i)~=0)
            presion(i)= length(pr)/ch;
            recall(i)=length(pr)/num(i);
            ndcg(i)=(sum(H(i,id2)./(log2((1:ch)+1))))/idcg(i);
             pq(an(i,1:m))=1:m;
            pg=pq(id1);
            ndcg1(i)=(sum(H(i,id1)./(log2(pg+1))))/idcg1(i);   
            mrr(i)=sum(H(i,id1)./pg);
           end
         end
        hit=sum(presion)/cao;
        reca=sum(recall)/cao;
        nd=sum(ndcg)/cao;
        nd1=sum(ndcg)/cao;
        mr=sum(mrr)/cao;
       fprintf('%d %d %d %d %d:S1:\titeration=%d\tpre5=%f\tre5=%f\tnd5=%f\tmrr=%f\n',ii,jj,kk,oo,zz,ep,hit,reca,nd,mr);
       
        if(hit>premax)
        premax=hit;
        remax=reca;
        ndmax=nd;
        nd1max=nd1;
        mrmax=mr;
      end

      

      %S2: prediction with confidence weights
      w=p*q';
      R=w.*R;
      clear w;
    
     loss=0;
        R(sub2ind(size(R),r1,r2))=-inf;
        [~,an]=sort(R,2,'descend');

        presion=zeros(n,1);
        recall=zeros(n,1);
         ndcg=zeros(n,1);
         ndcg1=zeros(n,1);
         mrr=zeros(n,1);
          num=sum(H,2);
        cao=full(sum(num~=0));                                       
         for i=1:n
           id1=find(H(i,:)~=0);
           id2=an(i,1:ch);
           pr=intersect(id1,id2);
           if(num(i)~=0)
            presion(i)= length(pr)/ch;
            recall(i)=length(pr)/num(i);
            ndcg(i)=(sum(H(i,id2)./(log2((1:ch)+1))))/idcg(i);
             pq(an(i,1:m))=1:m;
            pg=pq(id1);
            ndcg1(i)=(sum(H(i,id1)./(log2(pg+1))))/idcg1(i);   
            mrr(i)=sum(H(i,id1)./pg);
           end
         end
        hit=sum(presion)/cao;
        reca=sum(recall)/cao;
        nd=sum(ndcg)/cao;
        nd1=sum(ndcg1)/cao;
        mr=sum(mrr)/cao;
        fprintf('%d %d %d %d %d S2:\titeration=%d\tpre5=%f\tre5=%f\tnd5=%f\tmrr=%f\n',ii,jj,kk,oo,zz,ep,hit,reca,nd,mr);

        if(hit>premax)
        premax=hit;
        remax=reca;
        ndmax=nd;
        nd1max=nd1;
        mrmax=mr;
      end

       end
end

fprintf('%d %d %d %d %d:\t%d\t%f\t%f\t%f\n',ii,jj,kk,oo,zz,ep,premax,remax,ndmax);

 end
end
end
end
end