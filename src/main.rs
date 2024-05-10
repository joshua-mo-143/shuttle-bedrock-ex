use aws_config::Region;
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::{primitives::Blob, types::ResponseStream, Client};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum_streams::StreamBodyAs;
use futures::stream;
use serde::{Deserialize, Serialize};
use shuttle_runtime::SecretStore;

async fn hello_world() -> &'static str {
    "Hello, world!"
}

#[derive(Deserialize, Serialize)]
struct Prompt {
    prompt: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TitanResponse {
    input_text_token_count: i32,
    results: Vec<TitanTextResult>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
struct TitanTextResult {
    token_count: i32,
    output_text: String,
    completion_reason: String,
}

async fn prompt(
    State(state): State<AppState>,
    Json(Prompt { prompt }): Json<Prompt>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let titan_req = TitanRequest::new(prompt);
    let Ok(prompt) = serde_json::to_vec(&titan_req) else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let blob = Blob::new(prompt);

    let res = state
        .client
        .invoke_model()
        .body(blob)
        .model_id("amazon.titan-text-lite-v1:0:4k")
        .send()
        .await
        .unwrap();

    let res: &[u8] = &res.body.into_inner();
    let Ok(response_body) = serde_json::from_slice::<TitanResponse>(res) else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    let Some(TitanTextResult { output_text, .. }) = response_body.results.first() else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    Ok(output_text.to_owned())
}

async fn streamed_prompt(
    State(state): State<AppState>,
    Json(Prompt { prompt }): Json<Prompt>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let titan_req = TitanRequest::new(prompt);
    let Ok(message) = serde_json::to_vec(&titan_req) else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let blob = Blob::new(message);

    let res = state
        .client
        .invoke_model_with_response_stream()
        .body(blob)
        .model_id("amazon.titan-text-lite-v1:0:4k")
        .send()
        .await
        .unwrap();

    let stream = stream::unfold(res.body, |mut state| async move {
        let message = state.recv().await.unwrap();

        match message {
            Some(ResponseStream::Chunk(chunk)) => {
                let Ok(response_body) =
                    serde_json::from_slice::<TitanResponse>(&chunk.bytes.unwrap().into_inner())
                else {
                    println!("Unable to deserialize response body :(");
                    return None;
                };

                let Some(TitanTextResult { output_text, .. }) = response_body.results.first()
                else {
                    println!("No results :(");
                    return None;
                };

                Some((output_text.to_owned(), state))
            }
            _ => None,
        }
    });

    let stream = StreamBodyAs::text(stream);

    Ok(stream)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TitanRequest {
    input_text: String,
    text_generation_config: TextGenConfig,
}

impl TitanRequest {
    fn new(prompt: String) -> Self {
        Self {
            input_text: prompt,
            text_generation_config: TextGenConfig {
                temperature: 0.0,
                top_p: 0.0,
                max_token_count: 100,
                stop_sequences: vec!["|".to_string()],
            },
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TextGenConfig {
    temperature: f32,
    top_p: f32,
    max_token_count: i32,
    stop_sequences: Vec<String>,
}

#[derive(Clone)]
pub struct AppState {
    client: Client,
}

impl AppState {
    fn new(client: Client) -> Self {
        Self { client }
    }
}

async fn create_client(secrets: SecretStore) -> Client {
    let access_key_id = secrets
        .get("AWS_ACCESS_KEY_ID")
        .expect("AWS_ACCESS_KEY_ID not set in Secrets.toml");
    let secret_access_key = secrets
        .get("AWS_SECRET_ACCESS_KEY")
        .expect("AWS_ACCESS_KEY_ID not set in Secrets.toml");
    let aws_url = secrets
        .get("AWS_URL")
        .expect("AWS_ACCESS_KEY_ID not set in Secrets.toml");

    // note here that the "None" is in place of a session token
    let creds = Credentials::from_keys(access_key_id, secret_access_key, None);

    let cfg = aws_config::from_env()
        .endpoint_url(aws_url)
        .region(Region::new("eu-west-1"))
        .credentials_provider(creds)
        .load()
        .await;

    Client::new(&cfg)
}

#[shuttle_runtime::main]
async fn main(#[shuttle_runtime::Secrets] secrets: SecretStore) -> shuttle_axum::ShuttleAxum {
    let client = create_client(secrets).await;
    let appstate = AppState::new(client);
    let router = Router::new()
        .route("/", get(hello_world))
        .route("/prompt", post(prompt))
        .route("/prompt/streamed", post(streamed_prompt))
        .with_state(appstate);

    Ok(router.into())
}
