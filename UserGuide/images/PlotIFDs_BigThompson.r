rm(list = ls()) 
library(ggplot2)


# 24  hour-basic case
# Atlas 14 data:
CNREP6hr=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/IDFdata/CNREP-6h.txt',header=T,sep="\t")  
CNREP72hr=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/IDFdata/CNREP-72h.txt',header=T,sep="\t")  
USBR72hr=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/IDFdata/USBR_IDF.txt',header=T,sep="\t")


# standard catalog-smaller domain
# RainyDay data:
RD6hr=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_6hr/RainyDay_BigThompsonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD72hr=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_72hrCat/RainyDay_BigThompsonExample_72hrCat_FreqAnalysis.csv',header=T,sep=",")
# 72 hr
txtsize=14
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD72hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD72hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=rp,ymin=q5,ymax=q95),fill="red",data=CNREP72hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=rp,y=q50),data=CNREP72hr,size=1,color="red")+  
 
  geom_ribbon(aes(x=rp,ymin=q5,ymax=q95),fill="green",data=USBR72hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=rp,y=q50),data=USBR72hr,size=1,color="green")+  
   
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_72hr.png',width = 9, height = 6.5, units = "in",dpi=300)

# 6 hr
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD6hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD6hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=rp,ymin=q5,ymax=q95),fill="red",data=CNREP6hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=rp,y=q50),data=CNREP6hr,size=1,color="red")+  
  
  #geom_ribbon(aes(x=rp,ymin=q5,ymax=q95),fill="green",data=USBR72hr,alpha=0.25, linetype=2)+
  #geom_line(aes(x=rp,y=q50),data=USBR72hr,size=1,color="green")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,100),breaks=c(0,25,50,75,100),labels=c("0","25","50","75","100"))+
  coord_cartesian(ylim=c(0,100),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_6hr.png',width = 9, height = 6.5, units = "in",dpi=300)



# larger domain
# RainyDay data:
RD6hrLarge=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_6hrs_LargeDomain/RainyDay_BigThompsonExample_6hrs_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD72hrLarge=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_72hrs_LargeDomain/RainyDay_BigThompsonExample_72hrs_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
# 72 hr
txtsize=14
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD72hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD72hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD72hrLarge,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD72hrLarge,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_72hr_DomainComparison.png',width = 9, height = 6.5, units = "in",dpi=300)

# 6 hr
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD6hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD6hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD6hrLarge,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD6hrLarge,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,100),breaks=c(0,25,50,75,100),labels=c("0","25","50","75","100"))+
  coord_cartesian(ylim=c(0,100),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_6hrDomainComparison.png',width = 9, height = 6.5, units = "in",dpi=300)


# small domain, Stage IV
# RainyDay data:
RD6hrStageIV=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_6hr_StageIV/RainyDay_BigThompsonExample_6hr_StageIV_FreqAnalysis.csv',header=T,sep=",")
RD72hrStageIV=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/BigThompson/RainyDay_BigThompsonExample_72hr_StageIV/RainyDay_BigThompsonExample_72hr_StageIV_FreqAnalysis.csv',header=T,sep=",")
# 72 hr
txtsize=14
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD72hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD72hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD72hrStageIV,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD72hrStageIV,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,300),breaks=c(0,100,200,300),labels=c("0","100","200","300"))+
  coord_cartesian(ylim=c(0,300),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_72hr_StageIVComparison.png',width = 9, height = 6.5, units = "in",dpi=300)

# 6 hr
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD6hr,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD6hr,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD6hrStageIV,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD6hrStageIV,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(10,10000), breaks=c(10,20,50,100,200,500,1000,2000,5000,10000),labels=c("0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500","0.001\n1000","0.0005\n2000","0.0002\n5000","0.0001\n10000"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,100),breaks=c(0,25,50,75,100),labels=c("0","25","50","75","100"))+
  coord_cartesian(ylim=c(0,100),xlim=c(10,10000))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/BigThompson_6hrStageIVComparison.png',width = 9, height = 6.5, units = "in",dpi=300)



