
from django.urls import path
from . import views_new_final,views_for_hana_rag,report_generation

urlpatterns = [
    path('environment/create', views_new_final.create_environment, name='create_environment'),
    path('environment/<int:environment_id>', views_new_final.read_environment, name='read_environment'),
    path('environment/update/<int:environment_id>', views_new_final.update_environment, name='update_environment'),
    path('environment/delete/<int:environment_id>', views_new_final.delete_environment, name='delete_environment'),
    path('environments_by_email', views_new_final.get_all_environments_by_email, name='read_all_environments_by_mail'),
    path('environments', views_new_final.read_all_environments, name='read_all_environments'),
    path('agent/create', views_new_final.create_agent, name='create_agent'),
    path('agent/<int:agent_id>', views_new_final.read_agent, name='read_agent'),
    path('agent/update/<int:agent_id>', views_new_final.update_agent, name='update_agent'),
    path('agent/delete/<int:agent_id>', views_new_final.delete_agent, name='delete_agent'),
    path('agents_by_mail', views_new_final.get_all_agents_by_email, name='read_all_agents_by_mail'),
    path('agents/', views_new_final.read_all_agents, name='read_all_agents'),
    path('openai/env', views_new_final.create_openai_environment_api, name='openai_env_creation'),
    path('openai/run', views_new_final.run_openai_environment, name='openai_env_creation'),
   #path('send_email', views_new_final.send_email, name='sending email'),
    #path('openai/run_excel', views_new_final.run_openai_with_excel, name='openai_env_excel'),

    #HANA URLS
    path('processing_files', views_for_hana_rag.processing_files, name='processing_files'),
    path('query_making', views_for_hana_rag.query_system, name='query system'),

    #Dynamic agents
    path('dyn_create-agent', views_new_final.create_dynamic_agent, name='create_agent'),
    path('dyn_agents/<int:agent_id>', views_new_final.read_dynamic_agent, name='read_agent'),
    path('dyn_agents/<int:agent_id>/update', views_new_final.update_dynamic_agent, name='update_agent'),
    path('dyn_agents/<int:agent_id>/delete', views_new_final.delete_dynamic_agent, name='delete_agent'),
    path('dyn_agents_by_mail', views_new_final.get_all_dynamic_agents_by_email, name='read_all_dyn_agents_by_mail'),
    path('dyn_agents/', views_new_final.read_all_dynamic_agents, name='read_all_agents'),
    #path('create-openai-environment/', create_openai_environment_api, name='create_openai_environment'),
    path('run-agent-environment', views_new_final.run_agent_environment, name='run_agent_environment'),


   #Report Generation Url
    path('report_gen', report_generation.run_openai_environment_report, name='report_generation_api'),

]
