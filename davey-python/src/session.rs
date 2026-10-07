use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use std::num::NonZeroU16;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::signing_key_pair::SigningKeyPair;

/// Packets at least this large are encrypted or decrypted with the interpreter detached.
/// Below it (e.g. Opus audio frames) the work takes about a microsecond, and detaching would
/// cost more than it saves. The value matches CPython's `hashlib` threshold for releasing the GIL.
const DETACH_THRESHOLD_BYTES: usize = 2048;

/// Length and group size of the displayable code produced by `get_verification_code`.
const VERIFICATION_CODE_LENGTH: usize = 45;
const VERIFICATION_CODE_GROUP_SIZE: usize = 5;

/// Version of the key fingerprints a verification code is derived from.
const VERIFICATION_CODE_FINGERPRINT_VERSION: u16 = 0;

#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct CommitWelcome {
  pub commit: Vec<u8>,
  pub welcome: Option<Vec<u8>>,
}

impl From<davey::CommitWelcome> for CommitWelcome {
  fn from(cw: davey::CommitWelcome) -> Self {
    CommitWelcome {
      commit: cw.commit,
      welcome: cw.welcome,
    }
  }
}

// The session is shared between threads (e.g. an audio sender and an event loop), so every
// call takes the lock instead of relying on PyO3's per-object borrow flag.
// The core session stays boxed: it contains over-aligned fields, and Python only guarantees
// 8-byte alignment for object storage on 32-bit platforms.
#[pyclass(frozen)]
pub struct DaveSession {
  inner: Mutex<Box<davey::DaveSession>>,
}

impl DaveSession {
  /// Locks the session while staying attached to the interpreter.
  /// For short operations; waiting for the lock still detaches, so it cannot deadlock with
  /// the GIL or a stop-the-world pause.
  fn lock(&self, py: Python<'_>) -> MutexGuard<'_, Box<davey::DaveSession>> {
    // A panic in the core is raised as PanicException and poisons the lock. Keep the session
    // usable afterwards so it can still be reset or re-initialized.
    self
      .inner
      .lock_py_attached(py)
      .unwrap_or_else(PoisonError::into_inner)
  }

  /// Runs `f` on the locked session with the interpreter detached, for operations long enough
  /// to stall other Python threads.
  fn with_detached<T, F>(&self, py: Python<'_>, f: F) -> T
  where
    T: Send,
    F: FnOnce(&mut davey::DaveSession) -> T + Send,
  {
    py.allow_threads(|| f(&mut self.inner.lock().unwrap_or_else(PoisonError::into_inner)))
  }

  /// Runs `f` on the locked session, detaching only when `detach` is set.
  fn with_session<T, F>(&self, py: Python<'_>, detach: bool, f: F) -> T
  where
    T: Send,
    F: FnOnce(&mut davey::DaveSession) -> T + Send,
  {
    if detach {
      self.with_detached(py, f)
    } else {
      f(&mut self.lock(py))
    }
  }
}

/// Converts the Python-side key pair into the core type.
fn to_signing_key_pair(key_pair: Option<SigningKeyPair>) -> Option<davey::SigningKeyPair> {
  key_pair.map(|kp| davey::SigningKeyPair {
    private: kp.private,
    public: kp.public,
  })
}

#[pymethods]
impl DaveSession {
  #[new]
  #[pyo3(signature = (protocol_version, user_id, channel_id, key_pair=None))]
  fn new(
    py: Python<'_>,
    protocol_version: u16,
    user_id: u64,
    channel_id: u64,
    key_pair: Option<SigningKeyPair>,
  ) -> PyResult<Self> {
    let protocol_version =
      NonZeroU16::new(protocol_version).ok_or(py_value_error!("Unsupported protocol version"))?;

    let signing_key_pair = to_signing_key_pair(key_pair);

    let session = py
      .allow_threads(|| {
        davey::DaveSession::new(
          protocol_version,
          user_id,
          channel_id,
          signing_key_pair.as_ref(),
        )
      })
      .map_err(|e| py_value_error!("Failed to initialize session: {:?}", e))?;

    Ok(Self {
      inner: Mutex::new(Box::new(session)),
    })
  }

  #[pyo3(signature = (protocol_version, user_id, channel_id, key_pair=None))]
  fn reinit(
    &self,
    py: Python<'_>,
    protocol_version: u16,
    user_id: u64,
    channel_id: u64,
    key_pair: Option<SigningKeyPair>,
  ) -> PyResult<()> {
    let protocol_version =
      NonZeroU16::new(protocol_version).ok_or(py_value_error!("Unsupported protocol version"))?;

    let signing_key_pair = to_signing_key_pair(key_pair);

    self
      .with_detached(py, |session| {
        session.reinit(
          protocol_version,
          user_id,
          channel_id,
          signing_key_pair.as_ref(),
        )
      })
      .map_err(|err| py_value_error!("Failed to re-initialize session: {err:?}"))?;

    Ok(())
  }

  fn reset(&self, py: Python<'_>) -> PyResult<()> {
    self
      .with_detached(py, |session| session.reset())
      .map_err(|err| py_value_error!("Failed to reset session: {err:?}"))?;

    Ok(())
  }

  #[getter]
  fn protocol_version(&self, py: Python<'_>) -> u16 {
    self.lock(py).protocol_version().get()
  }

  #[getter]
  fn user_id(&self, py: Python<'_>) -> u64 {
    self.lock(py).user_id()
  }

  #[getter]
  fn channel_id(&self, py: Python<'_>) -> u64 {
    self.lock(py).channel_id()
  }

  #[getter]
  fn epoch(&self, py: Python<'_>) -> Option<u64> {
    self.lock(py).epoch().map(|e| e.as_u64())
  }

  #[getter]
  fn own_leaf_index(&self, py: Python<'_>) -> Option<u32> {
    self.lock(py).own_leaf_index().map(|e| e.u32())
  }

  #[getter]
  fn ciphersuite(&self, py: Python<'_>) -> u16 {
    self.lock(py).ciphersuite() as u16
  }

  #[getter]
  fn status(&self, py: Python<'_>) -> davey::SessionStatus {
    self.lock(py).status()
  }

  #[getter]
  fn ready(&self, py: Python<'_>) -> bool {
    self.lock(py).is_ready()
  }

  fn get_epoch_authenticator(&self, py: Python<'_>) -> Option<Vec<u8>> {
    self
      .lock(py)
      .get_epoch_authenticator()
      .map(|ea| ea.as_slice().to_vec())
  }

  #[getter]
  fn voice_privacy_code(&self, py: Python<'_>) -> Option<String> {
    self
      .lock(py)
      .voice_privacy_code()
      .map(|vpc| vpc.to_string())
  }

  fn set_external_sender(&self, py: Python<'_>, external_sender_data: &[u8]) -> PyResult<()> {
    self
      .with_detached(py, |session| {
        session.set_external_sender(external_sender_data)
      })
      .map_err(|err| py_value_error!("Failed to set external sender: {err:?}"))?;

    Ok(())
  }

  fn get_serialized_key_package(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
    let key_package = self
      .with_detached(py, |session| session.create_key_package())
      .map_err(|err| py_value_error!("Failed to create key package: {err:?}"))?;

    Ok(key_package)
  }

  #[pyo3(signature = (operation_type, proposals, expected_user_ids=None))]
  fn process_proposals(
    &self,
    py: Python<'_>,
    operation_type: davey::ProposalsOperationType,
    proposals: &[u8],
    expected_user_ids: Option<Vec<u64>>,
  ) -> PyResult<Option<CommitWelcome>> {
    let result = self
      .with_detached(py, |session| {
        session.process_proposals(operation_type, proposals, expected_user_ids.as_deref())
      })
      .map_err(|err| py_value_error!("Failed to process proposals: {err:?}"))?;

    Ok(result.map(CommitWelcome::from))
  }

  fn process_welcome(&self, py: Python<'_>, welcome: &[u8]) -> PyResult<()> {
    self
      .with_detached(py, |session| session.process_welcome(welcome))
      .map_err(|err| py_value_error!("Failed to process welcome: {err:?}"))?;

    Ok(())
  }

  fn process_commit(&self, py: Python<'_>, commit: &[u8]) -> PyResult<()> {
    self
      .with_detached(py, |session| session.process_commit(commit))
      .map_err(|err| py_value_error!("Failed to process commit: {err:?}"))?;

    Ok(())
  }

  // Equivalent to `davey::DaveSession::get_verification_code`, but only the key lookup holds
  // the lock; the scrypt hashing runs unlocked so it does not hold up encryption.
  fn get_verification_code(&self, py: Python<'_>, user_id: u64) -> PyResult<String> {
    let fingerprints = self
      .lock(py)
      .get_key_fingerprint_pair(VERIFICATION_CODE_FINGERPRINT_VERSION, user_id)
      .map_err(davey::errors::GetVerificationCodeError::from)
      .map_err(|e| py_value_error!("failed to generate verification code: {:?}", e))?;

    py.allow_threads(
      || -> Result<String, davey::errors::GetVerificationCodeError> {
        let output = davey::pairwise_fingerprints_internal(fingerprints)?;
        let code = davey::generate_displayable_code_internal(
          &output,
          VERIFICATION_CODE_LENGTH,
          VERIFICATION_CODE_GROUP_SIZE,
        )?;
        Ok(code)
      },
    )
    .map_err(|e| py_value_error!("failed to generate verification code: {:?}", e))
  }

  // Equivalent to `davey::DaveSession::get_pairwise_fingerprint`, but only the key lookup
  // holds the lock; the scrypt hashing runs unlocked so it does not hold up encryption.
  fn get_pairwise_fingerprint(
    &self,
    py: Python<'_>,
    version: u16,
    user_id: u64,
  ) -> PyResult<Vec<u8>> {
    let fingerprints = self
      .lock(py)
      .get_key_fingerprint_pair(version, user_id)
      .map_err(|e| py_value_error!("failed to generate pairwise fingerprint: {:?}", e))?;

    py.allow_threads(|| {
      davey::pairwise_fingerprints_internal(fingerprints)
        .map_err(davey::errors::GetPairwiseFingerprintError::from)
    })
    .map_err(|e| py_value_error!("failed to generate pairwise fingerprint: {:?}", e))
  }

  fn encrypt(
    &self,
    py: Python<'_>,
    media_type: davey::MediaType,
    codec: davey::Codec,
    packet: &[u8],
  ) -> PyResult<Vec<u8>> {
    let result = self
      .with_session(py, packet.len() >= DETACH_THRESHOLD_BYTES, |session| {
        session
          .encrypt(media_type, codec, packet)
          .map(|encrypted| encrypted.into_owned())
      })
      .map_err(|err| py_value_error!("Failed to encrypt: {err:?}"))?;

    Ok(result)
  }

  fn encrypt_opus(&self, py: Python<'_>, packet: &[u8]) -> PyResult<Vec<u8>> {
    self.encrypt(py, davey::MediaType::AUDIO, davey::Codec::OPUS, packet)
  }

  #[pyo3(signature = (media_type=None))]
  fn get_encryption_stats(
    &self,
    py: Python<'_>,
    media_type: Option<davey::MediaType>,
  ) -> Option<davey::EncryptionStats> {
    self
      .lock(py)
      .get_encryption_stats(media_type)
      .map(|s| s.to_owned())
  }

  fn decrypt(
    &self,
    py: Python<'_>,
    user_id: u64,
    media_type: davey::MediaType,
    packet: &[u8],
  ) -> PyResult<Vec<u8>> {
    let result = self
      .with_session(py, packet.len() >= DETACH_THRESHOLD_BYTES, |session| {
        session.decrypt(user_id, media_type, packet)
      })
      .map_err(|err| py_value_error!("Failed to decrypt: {err:?}"))?;

    Ok(result)
  }

  #[pyo3(signature = (user_id, media_type=None))]
  fn get_decryption_stats(
    &self,
    py: Python<'_>,
    user_id: u64,
    media_type: Option<davey::MediaType>,
  ) -> PyResult<Option<davey::DecryptionStats>> {
    let result = self
      .lock(py)
      .get_decryption_stats(user_id, media_type.unwrap_or(davey::MediaType::AUDIO))
      .map(|stats| stats.map(|s| s.to_owned()))
      .map_err(|err| py_value_error!("Failed to get decryption stats: {err:?}"))?;

    Ok(result)
  }

  fn get_user_ids(&self, py: Python<'_>) -> Vec<String> {
    self
      .lock(py)
      .get_user_ids()
      .map(|ids| {
        ids
          .into_iter()
          .map(|id| id.to_string())
          .collect::<Vec<String>>()
      })
      .unwrap_or_default()
  }

  fn can_passthrough(&self, py: Python<'_>, user_id: u64) -> bool {
    self.lock(py).can_passthrough(user_id)
  }

  #[pyo3(signature = (passthrough_mode, transition_expiry=None))]
  fn set_passthrough_mode(
    &self,
    py: Python<'_>,
    passthrough_mode: bool,
    transition_expiry: Option<u32>,
  ) {
    self
      .lock(py)
      .set_passthrough_mode(passthrough_mode, transition_expiry);
  }

  fn __repr__(&self, py: Python<'_>) -> String {
    let session = self.lock(py);
    format!(
      "<DaveSession protocol_version={}, user_id={}, channel_id={}, ready={}, status={:?}>",
      session.protocol_version(),
      session.user_id(),
      session.channel_id(),
      session.is_ready(),
      session.status()
    )
  }
}
